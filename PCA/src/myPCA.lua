--------------------------------------------------------------------------------
-- PCA with Torch7
--------------------------------------------------------------------------------
-- Koray Kavukcuoglu (https://github.com/koraykv/unsup)
-- Alfredo Canziani, Jul 14
--------------------------------------------------------------------------------

-- Instruction -----------------------------------------------------------------
-- This scripts aims to provide an understanding about how to play with PCA in
-- order to maniplate data dimensionality (for algorithmic speed or visualisa-
-- tion). Furthermore, it shows how to perform ZCA and PCA whitening.

-- Requires --------------------------------------------------------------------
require 'gnuplot'
require 'unsup'

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
print('eigenvectors (colums):'); print(v)
print('eigenvalues (power/variance):'); print(s)
print('sqrt of the above (energy/std):'); print(torch.sqrt(s))

-- Visualising -----------------------------------------------------------------
vv = v * torch.diag(torch.sqrt(s))
vv = torch.cat(torch.ones(2,1) * mean, vv:t())

X_hat = (X - torch.ones(m,1) * mean) * v[{ {},{1} }]

gnuplot.plot{
   {'dataset',X,'+'},
   {'PC1',vv[{ {1,1} , {} }],'v'},
   {'PC2',vv[{ {2,2} , {} }],'v'},
   {'reduced',X_hat:squeeze(), torch.zeros(m), '+'}
}
gnuplot.axis('equal')
gnuplot.axis{-20,50,-10,30}
gnuplot.grid(true)
