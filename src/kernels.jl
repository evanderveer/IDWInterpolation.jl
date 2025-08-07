module Kernel

gaussian(distance, shape_parameter) = exp(-shape_parameter * distance ^ 2)
inverse(distance) = 1 / distance
inverse_pow(distance, power) = 1 / (distance^power)

end #module