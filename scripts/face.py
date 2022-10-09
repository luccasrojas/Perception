import os
import time
import cv2 
import shutil
import ConsoleFormatter
import random
import rospkg
import face_recognition

from cv_bridge import CvBridge

PATH_PERCEPTION_UTLITIES = rospkg.RosPack().get_path('perception_utilities')
PATH_DATA = PATH_PERCEPTION_UTLITIES+'/resources/data/'

faceClassif = cv2.CascadeClassifier(PATH_PERCEPTION_UTLITIES+"/resources/model/haarcascade_frontalface_default.xml")
consoleFormatter=ConsoleFormatter.ConsoleFormatter()
bridge = CvBridge()


#Algorithm for face recognition
def recognize_face(req):
    print(consoleFormatter.format("\nRequested recognize_face_service", "WARNING"))
    threshold =req.threshold
    face_test_path= PATH_DATA+req.photo_name
    person =""
    result=""
    try:
        if os.path.exists(face_test_path):

            folder_face = PATH_DATA+'faces'

            for folders in os.listdir(folder_face):
                person_len_photo = len(os.listdir(folder_face+'/'+folders))
                if person_len_photo == 0:
                    os.rmdir(folder_face+'/'+folders)
                    print("The face "+folders+"was delete succesfuly")

            known_face_names = [None]*len(os.listdir(folder_face))
            cont = 0
            for names in known_face_names:
                known_face_names[cont] = os.listdir(folder_face)[cont]
                cont+=1

            current_face = 0
            nFaces = [None]*len(os.listdir(folder_face))

            photos_face = []
            known_faces = []
            know_face_encodings = []
                    
            for face_list in os.listdir(folder_face):

                specific_face_path= folder_face+'/'+face_list
                for img_list in os.listdir(folder_face+'/'+face_list):
                    photo_face=img_list
                    photos_face.append(photo_face)

                for current_photo in photos_face:
                    known_faces.append(face_recognition.load_image_file(specific_face_path+'/'+current_photo))


                for current_photo in known_faces:
                    #print(current_photo)
                    know_face_encodings.append(face_recognition.face_encodings(current_photo)[0])


                test_image = face_recognition.load_image_file(face_test_path)

                face_locations = face_recognition.face_locations(test_image)
                face_encodings = face_recognition.face_encodings(test_image,face_locations)

                matches = None
                name_of_the_face = "Unknow Person"
                for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(know_face_encodings,face_encoding)
                    #print(str(matches)+'->'+str(face_list))
                    n_trues = 0
                    for trues in matches:
                        if trues:
                            n_trues+=1
                    percent_true = n_trues/len(os.listdir(folder_face+'/'+face_list))
                    nFaces[current_face]= percent_true

                current_face+=1
                photos_face = []
                known_faces = []    
                know_face_encodings = []


            max_accert = 0
            if (max(nFaces)*100)>= threshold:
                max_accert = max(nFaces)
                index_max = nFaces.index(max_accert)
                name_of_the_face = known_face_names[index_max]
                summary = ""
                contador=-1
                for fresult in known_face_names:
                    contador +=1  
                    summary += " ["+ str(known_face_names[contador])+','+str(nFaces[contador])+']'
                    
            #print(consoleFormatter.format(summary,"OKBLUE"))
            print(consoleFormatter.format("Face name detected: "+ name_of_the_face+', '+str(max_accert*100)+'%',"OKBLUE"))
            print(consoleFormatter.format("Face recognition has been done successfully","OKGREEN"))



            person = name_of_the_face
            result = 'approved'
        else:
            print(consoleFormatter.format("The photo "+req.photo_name+" does not exist.","FAIL"))
            person = "NONE"
            result = 'not-approved'
    except:
        person = 'NONE'
        result = 'not-approved'
        print(consoleFormatter.format("It was not possible to recognize a person","FAIL"))
    return (person,result)


#Algorithm for saving a face
def save_face(utilities,req):
    try:
        umbral=10
        record_time = req.record_time
        name= req.name
        nPics = req.n_pics
        picsPath = PATH_DATA+"pics"
        facePath = PATH_DATA+"/faces"
        stop_record = False
        picsPersonPath = picsPath+"/"+name
        facePersonPath = facePath+"/"+name

        #Creates the folder where all the pictures are going to be saved
        if not os.path.exists(picsPersonPath):
                os.makedirs(picsPersonPath) 
        #Creates the folder of the faces in case it doesnt exist        
        if not os.path.exists(facePersonPath):
                        os.makedirs(facePersonPath)  
        siguiente = 0

        start =time.time()
        firstTime = True
        cont=0
        while stop_record!=True:
            cont+=1
            actual = time.time()
            if not os.path.exists(picsPersonPath):
                os.makedirs(picsPersonPath) 
            if(len(os.listdir(picsPersonPath)))>2000 or start+record_time<actual:
                stop_record = True
            #Image from the front camera
            #Posibilidad de cambiar esto a la imagen comprimida
            image_raw = utilities.image1
            cv2_img = bridge.imgmsg_to_cv2(image_raw, 'bgr8')
            largo = len(cv2_img[0])
            ancho = len(cv2_img)
            centro = largo//2

            faces = faceClassif.detectMultiScale(cv2_img, scaleFactor=1.3, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

            
            copy = cv2_img.copy()
            menor = 10000
            centinela = False
            outOfBounds =0
            centers = []
            for (x,y,w,h) in faces:
                factor_w = 0.5
                factor_h = 0.8
                scaled_w= int(w*(1+factor_w))
                scaled_h= int(h*(1+factor_h))
                #Factor scaling is better if faces are close or very far
                if x<factor_w*w/2 or y<factor_h*h/2 or x+w>largo-factor_w*w/2 or y+h>ancho-factor_h*h/2:
                    outOfBounds +=1
                    print("La cara se salió de la foto")
                    continue
                x-=int(factor_w*w/2)
                w=scaled_w
                y-=int(factor_h*h/2)
                h=scaled_h

                centroCara = (x+(x+w))//2
                if firstTime:
                    centroCaraAnterior = centroCara
                    firstTime= False
                #diferencia con el centro
                dif = centro-centroCara
                centers.append(dif)
                if centroCara > centroCaraAnterior-umbral and centroCara<centroCaraAnterior+umbral:
                    centroCaraAnterior=centroCara
                    centinela = True
                    # cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (0,255,0), 2)
                    rostro = copy[y:y+h, x:x+w]
                    rostro = cv2.resize(rostro, (150,180), interpolation=cv2.INTER_CUBIC)
            if centinela:                    
                siguiente+=1
                cv2.imwrite(picsPersonPath+"/"+name+str(siguiente)+".jpg",rostro)
                if centroCaraAnterior-centro<0:
                    menor = centro-centroCaraAnterior
                else:
                    menor = centroCaraAnterior-centro
                for diferencia in centers:
                    if diferencia-centro<0:
                        absDif = centro-diferencia
                    else:
                        absDif = diferencia-centro
                    #Se encontro una cara que se encuentra mas al centro que la anterior cara guardada    
                    if absDif < menor:
                        siguiente = 0
                        print(consoleFormatter.format("A face located nearer the center was found, restarting the save face process", "WARNING"))
                        centroCaraAnterior = diferencia
                        if os.path.isdir(picsPersonPath):
                            shutil.rmtree(picsPersonPath)
                    
        numSaved = nPics
        numPics = len(os.listdir(picsPersonPath))
        picsPerRange = numPics//numSaved

        #Saves random pictures from the pics file
        print(consoleFormatter.format("Saving random pictures from the saved ones", "WARNING"))
        for i in range(numSaved):
            pic = str(i*picsPerRange+random.randint(1,picsPerRange))
            shutil.copy2(picsPersonPath+"/{}{}.jpg".format(name,pic),facePersonPath+"/{}{}".format(name,pic)+".jpg")
        
        #Cleans the pictures for optimizing memory    
        if os.path.isdir(picsPersonPath):
            shutil.rmtree(picsPersonPath)

        print(consoleFormatter.format("The person: {} has been saved".format(name), "OKGREEN"))
        return True

    except:
        if os.path.isdir(picsPersonPath):
            shutil.rmtree(picsPersonPath)
            shutil.rmtree(facePersonPath)      
        print(consoleFormatter.format("The image coudn't be saved", "FAIL"))
        return False   

    # def callback_save_face_srv(self, req):
    #     print(consoleFormatter.format("\nRequested save face service", "WARNING"))
    #     #try:
    #     picsPath = self.PATH_DATA+"pics"
    #     if not os.path.exists(picsPath+"/{}".format(req.name)):
    #             os.makedirs(picsPath+"/{}".format(req.name))    
    #     # if not self.isFrontCameraUp:
    #     #     turn_camera = turn_camera_srvRequest()
    #     #     turn_camera.camera_name = self.CAMERA_FRONT
    #     #     turn_camera.enable = "enable"
    #         # self.callback_turn_camera_srv(turn_camera)
    #     stop_record = False
    #     facePath = self.PATH_DATA+"/faces"
    #     if not os.path.exists(facePath+"/{}".format(req.name)):
    #             os.makedirs(facePath+"/{}".format(req.name))    
    #     personPath = picsPath+"/"+req.name
    #     personPath1 = facePath+"/"+req.name
    #     if not os.path.exists(personPath):
    #             os.makedirs(personPath)
    #     siguiente = 0
    #     if len(os.listdir(personPath))!=0:
    #         for imageName in os.listdir(personPath):
    #             if int(imageName.replace(".jpg","")[-1])>siguiente:
    #                 siguiente = int(imageName.replace(".jpg","")[-1])  
    #     if len(os.listdir(personPath1))!=0:
    #         for imageName in os.listdir(personPath1):
    #             if int(imageName.replace(".jpg","")[-1])>siguiente:
    #                 siguiente = int(imageName.replace(".jpg","")[-1])  
    #     start_time = time.time()
    #     promedio = []            
    #     while len(promedio)<100 and not stop_record:
    #         stop_time = time.time()
    #         image_raw = self.image1
    #         cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
    #         largo = len(cv2_img[0])
    #         centro = largo//2
    #         faces = self.faceClassif.detectMultiScale(cv2_img, 1.3, 5)
    #         copy = cv2_img.copy()
    #         menor = 10000
    #         dif = None
    #         difabs = None
    #         for (x,y,w,h) in faces:
    #             if x<20 or y<20 or x+w>largo-20 or y+h>centro-20:
    #                 continue
    #             x-=20
    #             w+=40
    #             y-=20
    #             h+=40
    #             centroCara = (x+(x+w))//2
    #             print("El centro de la cara es",centroCara)
    #             #diferencia con el centro
    #             difabs = centro-centroCara
    #             if centroCara<centro:
    #                 difabs = centro-centroCara
    #             else:
    #                 difabs = centroCara-centro
    #         if len(faces)!=0 or difabs is not None:
    #             promedio.append(difabs)        
    #         if(stop_time-start_time)>10.0:
    #             stop_record = True   
    #     if stop_record:
    #         print(consoleFormatter.format("No face detected ", "FAIL"))
    #         if os.path.isdir(picsPath+"/{}".format(req.name)):
    #             shutil.rmtree(picsPath+"/{}".format(req.name))
    #             shutil.rmtree(facePath+"/{}".format(req.name))  
    #         return False    
    #     avg = mean(promedio)
    #     start_time = time.time()
    #     while stop_record!=True:
    #         stop_time = time.time()
    #         image_raw = self.image1
    #         cv2_img = self.bridge.imgmsg_to_cv2(image_raw, 'bgr8')
    #         largo = len(cv2_img[0])
    #         centro = largo//2
    #         print("EL centro de la imagen es:",centro)
    #         faces = self.faceClassif.detectMultiScale(cv2_img, 1.3, 5)
    #         copy = cv2_img.copy()
    #         menor = 10000
    #         centinela = False
    #         for (x,y,w,h) in faces:
    #             if x<20 or y<20 or x+w>largo-20 or y+h>centro-20:
    #                 print("La cara se salió de la foto")
    #                 continue
    #             x-=20
    #             w+=40
    #             y-=20
    #             h+=40
    #             centroCara = (x+(x+w))//2
    #             #print("El centro de la cara es",centroCara)
    #             #diferencia con el centro

    #             dif = centro-centroCara

    #             if centroCara<centro:
    #                 absdif = centro-centroCara
    #             else:
    #                 absdif = centroCara-centro
    #             #print("La diferencia es:",dif)
    #             if absdif<menor and dif > avg-40 and dif < avg+40:
    #                 centinela = True
    #                 cv2.rectangle(cv2_img, (x,y), (x+w, y+h), (0,255,0), 2)
    #                 rostro = copy[y:y+h, x:x+w]
    #                 rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
    #                 menor = absdif
    #         if centinela:
    #             siguiente+=1
    #             cv2.imwrite(personPath+"/"+req.name+str(siguiente)+".jpg",rostro)
    #         if(stop_time-start_time)>req.record_time:
    #             stop_record = True
    #     numSaved = req.n_pics
    #     numPics = len(os.listdir(picsPath+"/{}".format(req.name)))
    #     picsPerRange = numPics//numSaved
    #     for i in range(numSaved):
    #         pic = str(i*picsPerRange+random.randint(1,picsPerRange))
    #         shutil.copy2(personPath+"/{}{}.jpg".format(req.name,pic),personPath1+"/{}{}".format(req.name,pic)+".jpg")
    #     print(consoleFormatter.format("The person: {} has been saved".format(req.name), "OKGREEN"))
    #     print(consoleFormatter.format("El promedi fue de: {}".format(avg), "OKGREEN"))
    #     return True
    #     #except:
    #         # if os.path.isdir(picsPath+"/{}".format(req.name)):
    #         #     shutil.rmtree(picsPath+"/{}".format(req.name))
    #         #     shutil.rmtree(facePath+"/{}".format(req.name))      
    #         # print(consoleFormatter.format("The image coudn't be saved", "FAIL"))
    #         # return False