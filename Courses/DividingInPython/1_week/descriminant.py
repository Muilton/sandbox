import sys
a = int(sys.argv[1])
b = int(sys.argv[2])
c = int(sys.argv[3])

x1 = int((-b - (b**2 - 4*a*c)**0.5) / (2*a))
x2 = int((-b + (b**2 - 4*a*c)**0.5) / (2*a))

print(x1)
print(x2)