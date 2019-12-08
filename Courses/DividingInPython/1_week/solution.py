import sys
digit_string = sys.argv[1]

def sum(x):
    result = 0
    for i in digit_string:
        result = result + int(i) 
    print(result)
    return result

sum(digit_string)