marks=[78,85,90,66,88]
total=sum(marks)
average=total/len(marks)
maximum=max(marks)
print("Total Marks:",total)
print("Average Marks:",average)
print("Higest Marks:",maximum)

for average in marks:
    if average>=90 and average<=100:
        print("Grade: A")
    elif average>=80 and average<90:
        print("Grade: B")
    elif average>=70 and average<80:
        print("Grade: C")
    elif average>=60 and average<70:
        print("Grade: D")
    else:
        print("Grade: Fail")