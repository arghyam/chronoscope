annotation_prompt = """
i want to create a streamlit application which does the follwoing task
1. Ask for the suer to put the path of the image sthat are there
2. Once the user puts out the path of the image folder , it will show the image one by one 
3. The user has to annotate . so there would be 6 options out of which the user has to clck  one of them . they are : 
Good
Blurry
Out of focus
Oriented
Foggy and 
Poor lighting
4. Once an image gets annoatated , it goes to the next image accordingly. 
5. Also a csv file has to be mantained where the annotation should be saved. these things needs to be saved. They are :
Image name , annotation out of 6 types what was the annotation that the user clicked , annotation done ( yes/no)
6. finally lets say i open up this application again and the folder that i have chosen , it should start from the point that i left off for the remaining annotation left. 
"""