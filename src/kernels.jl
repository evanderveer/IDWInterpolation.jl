module Kernels

abstract type Kernel end

struct Gaussian <: Kernel
    shape_parameter
end
(k::Gaussian)(x) = exp(-k.shape_parameter * x ^ 2)

struct Inverse <: Kernel
end
(k::Inverse)(x) = 1 / x

struct InversePow <: Kernel
    power
end
(k::InversePow)(x) = 1 / (x ^ k.power)

struct Constant <: Kernel
end
(k::Constant)(x) = 1

end #module