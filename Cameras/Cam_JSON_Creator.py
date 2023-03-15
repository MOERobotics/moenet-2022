import json

def fillout_json(info : dict):

    #Check if all values are writeable
    writeable = True
    for value in info.values():
        if type(value) == dict:
            writeable = False
            break
    
    if writeable:
        while True:
            info_len = len(info)
            #Print what is being edited
            print('Editing the follwing values: ', end = '')
            print(*info.keys(), sep = ', ')
            confirm = input(f'Would you like to write all {info_len} values at once?: ')
            confirm = confirm.lower()

            if len(confirm) == 0:
                continue

            if confirm[0] == 'y':
                keylist = list(info.keys())
                vallist = list(info.values())
                print('Input them as comma separated values:')

                while True:
                    vals = input()
                    vals = [i.strip() for i in vals.strip().split(sep = ',')]

                    if len(vals) != info_len:
                        print(f'Not enough variables provided. Needed {info_len}, given {len(vals)}.')
                        continue
                    
                    for i in range(info_len):
                        try:
                            vals[i] = type(vallist[i])(vals[i])
                            info[keylist[i]] = vals[i]
                        except:
                            print(f'Wrong type of value provided for {keylist[i]}. Given {vals[i]} of {type(vals[i])} but need {type(vallist[i])}.')
                            break
                    else:
                        break

                return
            elif confirm[0] == 'n':
                break
            else:
                print('Invalid input')

    for key in info.keys():
        if type(info[key]) == dict:
            print(f'Filling out values for {key}:')
            fillout_json(info[key])
        else:
            while True:
                val = input(f'{key}: ')

                if val == '':
                    print('Please enter a value')
                    continue

                try:
                    val = type(info[key])(val)
                    info[key] = val
                    break
                except:
                    print(f'Wrong type of value provided for {key}. Given {val} of {type(val)} but need {type(info[key])}.')


#Valid file formats have to follow the format Cam_*.json. This stores *

info = {
    #Serial Number
    'MXID': '',
    #Lite or S2
    'Type': '',
    #tag, obj, or hybrid
    'Mode': '',
    'Pose': {
        'Translation' : {
            'x' : 0.0,
            'y' : 0.0,
            'z' : 0.0,
        },
        'Rotation' : {
            'w' : 0.0,
            'i' : 0.0,
            'j' : 0.0,
            'k' : 0.0,
        }
    }
}


print('Please fill out the following information: ')
camera_name = input("Enter the camera name: ")
fillout_json(info)

with open(f"Cam_{camera_name}.json", 'w') as output:
    json_obj = json.dumps(info, indent = 4)
    output.write(json_obj)

print('Finished writing')