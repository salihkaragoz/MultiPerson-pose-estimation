require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')
-------------------
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end
--------------------
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end
-----------------------
preds = torch.Tensor(273600,17,2)
local j = 0
local file = 'output_of_test2017zerothre03.txt'
m = torch.load('model_125.t7')
local lines = lines_from(file)
t = {}
local i = 0
------------------------

for k,v in pairs(lines) do

  local msa = v
  local i = 0
  for w in msa:gmatch("%S+") do
    local msk = w    
    t[i] = msk
    i = i +1 
  end
  local im = image.load("/home/muhammed/developments/SHG/pose-hg-demo/test2017/"..t[0])
  
  t[2] = tonumber(t[2])
  t[3] = tonumber(t[3])
  t[4] = tonumber(t[4])
  t[5] = tonumber(t[5])
  local center = {(t[4]+t[2])/2 ,(t[5]+t[3])/2}
  scale = t[5]-t[3]
  scale = scale  / 200
  local inp = crop(im, center, scale, 0, 256)
  local out = m:forward(inp:view(1,3,256,256):cuda())
  
  cutorch.synchronize()
  
  local hm = out[#out][1]:float()
  
  hm[hm:lt(0)] = 0
  --print(hm)
  print(j.."       Count")
  -- Get predictions (hm and img refer to the coordinate space)
  local preds_hm, preds_img = getPreds(hm, center, scale)
  j = j + 1
  preds[j]:copy(preds_img)
  
 
  print(preds_img)
  --print(preds)



  collectgarbage()  
end

--print(preds)
--local predFile = hdf5.open('preds/output_of_val2017.h5', 'w')
--predFile:write('preds', preds)
--predFile:close()
