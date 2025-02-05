
print(" The Prime Numbers within first 100 integers are: \n")

for num in range(2, 100):
    if num == 2:
        print(num, end=" ")
    else:
        div=2
        is_prime = True
        for div in range(div,num):
            if num%div == 0:
                is_prime = False
                break
        if(is_prime== True):
            print(num, end=" ")




