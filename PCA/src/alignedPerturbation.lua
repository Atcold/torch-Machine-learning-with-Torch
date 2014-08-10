--------------------------------------------------------------------------------
-- Data augmentation by aligned perturbation
--------------------------------------------------------------------------------
-- Alfredo Canziani, Aug 14
--------------------------------------------------------------------------------

-- Instruction -----------------------------------------------------------------
-- This scripts aims to provide an understanding about how to play with PCA in
-- order to generate plausible fake data for fighting overfitting.
-- No user input is required, but you are very welcome to muck around.

-- Requires --------------------------------------------------------------------
require 'image'
require 'gnuplot'
require 'sys'

-- Function definition (skip, not important) -----------------------------------
function rgb(rgb)
   return rgb[1]*256^2 + rgb[2]*256^1 + rgb[3]*256^0
end

function dumpToFile(colourImage)
   data = io.open('aux/dataPoints.dat','w+')
   for i = 1, colourImage:size(2) do
      for j = 1, colourImage:size(3) do
         data:write(string.format(
            '%f %f %f %d\n',
            colourImage[1][i][j],
            colourImage[2][i][j],
            colourImage[3][i][j],
            rgb(colourImage[{ {},i,j }])
         ))
      end
   end
   data:close()
end

function gnuplot.colourPxDistribution(image)
   dumpToFile(image)
   plotCmd = io.open('aux/plot3D.plt','r')
   gnuplot.raw(plotCmd:read('*all'))
   plotCmd:close()
   gnuplot.title('3D colourspace pixels distribution')
end

function image.loadByte(str)
   return image.load(str):mul(255):add(.5):floor()
end

-- Loading dataset/image -------------------------------------------------------
-- Load image in byte (0-255) format
img = image.loadByte('aux/peppers.png')

-- Display the image and the px distribution
image.display{image = img, zoom = 4, legend = 'Original image', min = 0, max = 255}
gnuplot.colourPxDistribution(img)

-- Rearranging pixel components along 3-column X matrix
imgT = img:transpose(1,2):transpose(2,3):clone()
X = imgT:reshape(img:size(2)*img:size(3),img:size(1))

-- PCA -------------------------------------------------------------------------
-- X is m x n
mean = torch.mean(X, 1) -- 1 x n
m = X:size(1)
Xm = X - torch.ones(m, 1) * mean
Xm:div(math.sqrt(m - 1))
v,s,_ = torch.svd(Xm:t())
s:cmul(s) -- n

-- v: eigenvectors, s: eigenvalues of covariance matrix
b = sys.COLORS.blue; n = sys.COLORS.none
print(b .. 'eigenvectors (columns):' .. n); print(v)
print(b .. 'sqrt of eigenvalues (energy/std):' .. n); print(torch.sqrt(s))

-- Scaling eigenvectors with corresponding std
vv = v * torch.diag(torch.sqrt(s))

-- Visualising PCA -------------------------------------------------------------
-- Line style for PCA arrows
gnuplot.raw('set style line 53 lt 1 lc rgb "white" lw 3')

-- Drawing arrows
mean = mean:squeeze()
for cmp = 1, 3 do
   arrow = mean + vv[{ {},cmp }]
   cmd = string.format(
      'set arrow %d from %f,%f,%f to %f,%f,%f empty ls 53 front',
      3 + cmp,
      mean [1], mean [2], mean [3],
      arrow[1], arrow[2], arrow[3]
   )
   gnuplot.raw(cmd)
end
gnuplot.plotflush()

-- Aligned perturbation --------------------------------------------------------
collection1 = {}
for i = 1, 12 do
   perturbation = vv * torch.randn(3,1) * 0.2
   X_hat = X + torch.ones(m,1) * perturbation:t()
   img_hatT = X_hat:reshape(img:size(2),img:size(3),img:size(1))
   img_hat = img_hatT:transpose(2,3):transpose(1,2)
   table.insert(collection1,img_hat:clone())
end
image.display{
   image = collection1, legend = 'Aligned perturbation',
   zoom = 4/3, nrow = 4, min = 0, max = 255
}

-- Disaligned perturbation -----------------------------------------------------
collection2 = {}
for i = 1, 12 do
   perturbation = torch.randn(3,1) * 0.2 * math.sqrt(s:sum())
   X_hat = X + torch.ones(m,1) * perturbation:t()
   img_hatT = X_hat:reshape(img:size(2),img:size(3),img:size(1))
   img_hat = img_hatT:transpose(2,3):transpose(1,2)
   table.insert(collection2,img_hat:clone())
end
image.display{
   image = collection2, legend = 'Disaligned  perturbation',
   zoom = 4/3, nrow = 4, min = 0, max = 255
}
