-- paper.md -> LaTeX. A paragraph `{{name}}` becomes latex/blocks/name.tex (or `\name`, e.g. {{clearpage}}), and the
-- caption paragraph after it fills the block's %CAPTION%. "Abstract" wraps the next paragraph; "A.N ..." headings get app:aN.
local stringify = pandoc.utils.stringify

local function latex(block)
  return (pandoc.write(pandoc.Pandoc({block}), 'latex', {cite_method = 'natbib'}):gsub('%s+$', ''))
end

function Header(h)
  local n = stringify(h):match('^A%.(%d+) ')
  if n then h.identifier = 'app:a' .. n end
  return h
end

function Pandoc(doc)
  local b, out, i = doc.blocks, {}, 1
  while i <= #b do
    local name = b[i].t == 'Para' and stringify(b[i]):match('^{{([%w-]+)}}$')
    if name then
      local f = io.open('latex/blocks/' .. name .. '.tex')
      local tex = f and f:read('a') or '\\' .. name
      if tex:find('%CAPTION%', 1, true) then
        i = i + 1
        local caption = latex(b[i])
        tex = tex:gsub('%%CAPTION%%', function() return caption end)
      end
      out[#out + 1] = pandoc.RawBlock('latex', tex)
    elseif b[i].t == 'Header' and stringify(b[i]) == 'Abstract' then
      i = i + 1
      out[#out + 1] = pandoc.RawBlock('latex', '\\begin{abstract}\n' .. latex(b[i]) .. '\n\\end{abstract}')
    else
      out[#out + 1] = b[i]
    end
    i = i + 1
  end
  doc.blocks = out
  return doc
end
