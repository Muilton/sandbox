import sys
num_steps = int(sys.argv[1])

def draw(steps):
    for i in range(steps, 0, -1):
        print(" "*(i-1)+"#"*(steps-(i-1)))

draw(num_steps)
