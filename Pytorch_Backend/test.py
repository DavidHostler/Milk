import os 
current_directory = os.getcwd()
folder_list = current_directory.split('/')
root_dir= ''
#Get path of root directory of entire project
for i in range(len(folder_list)-1): 
    if i < len(folder_list)-1: #This makes sure the slash is added correctly
        root_dir+= folder_list[i] + '/'
    else:
        root_dir+= folder_list[i]
print(dir, len(os.listdir(dir)))
# print(dir, current_directory)
# print(root_dir== current_directory)