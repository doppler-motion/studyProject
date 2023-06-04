"""
计算购买苹果和橘子的总价，以及总价相对于各参数的导数

"""

from layer_native import MulLayer, AddLayer

apple_num = 2
apple = 100
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_orange_apple_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
apple_orange_price = add_orange_apple_layer.forward(apple_price, orange_price)  # (3)
final_price = mul_tax_layer.forward(apple_orange_price, tax)  # (4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) # (4)
dapple_price, dorange_price = add_orange_apple_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(final_price)
print(dapple_num, dapple, dorange_num, dorange, dtax)

