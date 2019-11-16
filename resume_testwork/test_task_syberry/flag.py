class ArgumentError(Exception):
    pass


def flag(N=None):
    if not N or not isinstance(N, int) or N <= 0 or not N % 2 == 0:  # check input value. if not - raise Error
        raise ArgumentError(
            f"Input value isn't correct: {N}  {type(N)}\n Please, use only a positive integer even number")

    height = 2 * N + 1
    weight = 3 * N + 1
    upper = N / 2  # upper shift
    left = N  # left shift
    inner = 1  # inner shift for fill out by 0
    matrix = [[0 for x in range(weight + 1)] for y in range(height + 1)]  # create a 2-D array like filed
    result = ""  # string with result to return

    for i in range(len(matrix)):
        if i <= N:
            shift = N - i  # shift of upper part of the rhombus
        else:
            shift = i - N - 1  # shift of bottom part of the rhombus

        # to walk on an every element of the 2-d array and replace marker if it needs
        for j in range(len(matrix[i])):
            if i == 0 or i == height or j == 0 or j == weight:
                matrix[i][j] = "#"  # draw border of area
            elif (i > upper + inner) and (i < height - upper - inner) and (j > left + shift + inner) and (
                    j < weight - left - shift - inner):
                matrix[i][j] = "0"  # draw inner part of the rhombus
            elif (i > upper) and (i < height - upper) and (j > left + shift) and (j < weight - left - shift):
                matrix[i][j] = "*"  # draw border of the rhombus
            else:
                matrix[i][j] = " "

    for i in matrix:  # concatenate the matrix in a line with "\n" like line separator
        result += "".join(i) + "\n"

    return result

