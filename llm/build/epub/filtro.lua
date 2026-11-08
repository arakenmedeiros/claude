local heads = {}
for line in io.lines("/tmp/epub/heads.tsv") do
  local lv, tx = line:match("^(%d)\t(.*)$")
  if lv then table.insert(heads, {tonumber(lv), tx}) end
end
local ptr = 1
local function norm(s) return (s:gsub("%s+", " "):gsub("^ ", ""):gsub(" $", "")) end
local NB = "\u{00A0}"
function Table(tb)
  if #tb.bodies ~= 1 then return nil end
  local rows = tb.bodies[1].body
  if #rows ~= 1 or #rows[1].cells ~= 1 then return nil end
  local blocks = rows[1].cells[1].contents
  local has = false
  for _, b in ipairs(blocks) do if pandoc.utils.stringify(b):find(NB, 1, true) then has = true break end end
  if not has then return nil end
  local out, lines, cls = {}, {}, nil
  for k, b in ipairs(blocks) do
    local t = pandoc.utils.stringify(b)
    if k == 1 and not t:find(NB, 1, true) then
      table.insert(out, pandoc.Para({pandoc.Strong(b.content)}))
      if t:upper():find("PYTHON") then cls = "python" end
    else
      table.insert(lines, (t:gsub(NB, " ")))
    end
  end
  table.insert(out, pandoc.CodeBlock(table.concat(lines, "\n"), pandoc.Attr("", cls and {cls} or {})))
  return out
end
function Para(el)
  if ptr > #heads then return nil end
  local t = norm(pandoc.utils.stringify(el))
  for k = ptr, math.min(ptr + 6, #heads) do
    if t == norm(heads[k][2]) then
      ptr = k + 1
      return pandoc.Header(heads[k][1], el.content)
    end
  end
  return nil
end
function Pandoc(doc)
  local out, skipping = {}, false
  for _, b in ipairs(doc.blocks) do
    if b.t == "Para" and norm(pandoc.utils.stringify(b)) == "Sumário" and #out < 40 then skipping = true end
    if b.t == "Header" then skipping = false end
    if not skipping then table.insert(out, b) end
  end
  doc.blocks = out
  io.stderr:write("cabeçalhos casados: " .. (ptr - 1) .. " de " .. #heads .. "\n")
  return doc
end
return {{Table = Table, Para = Para}, {Pandoc = Pandoc}}
