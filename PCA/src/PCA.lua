--------------------------------------------------------------------------------
-- PCA with Torch7
--------------------------------------------------------------------------------
-- Koray Kavukcuoglu (https://github.com/koraykv/unsup)
-- Alfredo Canziani, Jul, Aug 14
--------------------------------------------------------------------------------

-- Instruction -----------------------------------------------------------------
-- This scripts aims to provide an understanding about how to play with PCA in
-- order to maniplate data dimensionality (for algorithmic speed or visualisa-
-- tion). Furthermore, it shows how to perform ZCA and PCA whitening.

-- Choose a spherification radius (in normal application is set to 1, but bigger
-- values will look better in the chart and enable ZCA display
--<<<
radius = 3 -- 1, 3 for visualisation sake
showZCA = false -- true/false
-->>>

-- You want, perhaps, also to try to enable and disable the visualisation of the
-- rotated data and PCA whitening in the ZCA visualisation section below (line
-- 84 and below)

-- Requires --------------------------------------------------------------------
require 'gnuplot'
require 'unsup'
require 'sys'

-- Define dataset --------------------------------------------------------------
-- Random 2D data with std ~(1.5,6)
N = 100
math.randomseed(os.time())
x1 = torch.randn(N) * 1.5 + math.random()
x2 = torch.randn(N) * 6 + 2 * math.random()
X = torch.cat(x1, x2, 2) -- Nx2

-- Rotating the data randomly
theta = math.random(180) * math.pi / 180
R = torch.Tensor{
   {math.cos(theta), -math.sin(theta)},
   {math.sin(theta),  math.cos(theta)}
}
X = X * R:t()
X[{ {},1 }]:add(25)
X[{ {},2 }]:add(10)

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
print(b .. 'eigenvalues (power/variance):' .. n); print(s)
print(b .. 'sqrt of the above (energy/std):' .. n); print(torch.sqrt(s))

-- Projection ------------------------------------------------------------------
X_hat = (X - torch.ones(m,1) * mean) * v[{ {},{1} }] -- m x 1

-- Visualising PCA -------------------------------------------------------------
vv = v * torch.diag(torch.sqrt(s))
vv = torch.cat(torch.ones(2,1) * mean, vv:t())

gnuplot.plot{
   {'dataset',X,'+'},
   {'PC1',vv[{ {1,1} , {} }],'v'},
   {'PC2',vv[{ {2,2} , {} }],'v'},
   {'reduced',X_hat:squeeze(), torch.zeros(m), '+'}
}
gnuplot.axis('equal')
gnuplot.axis{-20,50,-10,30}
gnuplot.grid(true)

-- ZCA / spherification / whitening --------------------------------------------
X_rot = (X - torch.ones(m,1) * mean) * v
X_PCA_white = X_rot * torch.sqrt(s):pow(-1):mul(radius):diag()
X_ZCA_white = X_PCA_white * v:t()

-- Visualising ZCA -------------------------------------------------------------
if showZCA then
   gnuplot.figure(2)
   gnuplot.plot{
      {'dataset',X,'+'},
--    {'rortated',X_rot,'+'},
--    {'PCA white',X_PCA_white,'+'},
      {'ZCA white',X_ZCA_white,'+'}
   }
   gnuplot.axis('equal')
   gnuplot.axis{-20,50,-10,30}
   gnuplot.grid(true)
end
