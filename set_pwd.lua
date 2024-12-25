local handle = io.popen("pwd") -- 在类 Unix 系统上
if (handle == nil) then
    print("Error to set handle")
    return
end
local current_dir = handle:read("*a"):gsub("%s+$", "") -- 去除末尾的换行符
handle:close()
local test_json_dir = current_dir.."/test/json-test"

local function replace_loader_in_file(filepath, new_content)
    -- 打开文件以读取模式
    local file = io.open(filepath, "r")
    if not file then
        print("Error: Cannot open file " .. filepath)
        return
    end

    -- 读取文件内容
    local content = file:read("*a")
    file:close()

    -- 使用 gsub 替换所有 loader(...) 的内容
    local updated_content = content:gsub("loader%([^%)]*%)", "loader(" .. new_content .. ")")

    -- 如果内容未发生变化，直接返回
    if content == updated_content then
        print("No changes made to " .. filepath)
        return
    end

    -- 以写模式重新打开文件
    file = io.open(filepath, "w")
    if not file then
        print("Error: Cannot open file for writing " .. filepath)
        return
    end

    -- 写入修改后的内容
    file:write(updated_content)
    file:close()

    print("Updated file: " .. filepath)
end

local function traverse_dir(test_dir, json_dir)
    local p = io.popen("ls -A " .. test_dir)
    if p == nil then
        print("Error")
        return
    end
    for file in p:lines() do
        local base_name = file:match("(.+)-test%.cc")
        if base_name == nil or base_name == "json" then
            goto continue
        end
        local full_test_path = test_dir .. "/" .. file
        local full_json_path = json_dir.."/"..base_name..".json"
        replace_loader_in_file(full_test_path, '"'..full_json_path..'"')
        ::continue::
    end
end

traverse_dir(current_dir.."/test", test_json_dir)
