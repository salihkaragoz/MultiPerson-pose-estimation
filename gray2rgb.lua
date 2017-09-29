function gray2rgb(im)
	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
	if dim == 3 then
		 print('Expected 1 channel')
		 return im
	end

	local rgb = torch.zeros(3, w, h)

	rgb[1] = im[1]
	rgb[2] = im[1]
	rgb[3] = im[1]

	return rgb
end
