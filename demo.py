n = int(input())
for i in range(n):
    for j in range(n):
        print((((j + 1) * 2 - 1 ) if ( i + j < n) else ((n - i) * 2 - 1)) if (i >= j) else (((i + 1) * 2) if (i + j < n) else ((n - j) * 2)), end=' ')
    print()