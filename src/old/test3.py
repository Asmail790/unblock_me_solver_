x1 = 0.3
x2 = 0.7

def toL(y:float)->int:

    values = [x1+(x2-x1)/6*z for z in range(7)] 
    print(values)

    for i in range(len(values)-1):
        if values[i] <= y <= values[i+1]:
            return i 
    
    if  y < values[0]:
        return 0 
    
    elif values[-1]< y:
        return len(values) - 2 
    
    else:
        raise ValueError()