import model_train as mt
import sys
from sklearn.metrics import accuracy_score, f1_score

def usage():
    print("Usage\nrun | model_to_test | class_1 class_2 ... class_n \nexample: run decision_tree feature_envy god_class"
          "\ncompare | model_to_test | class\nexample: compare decision_tree feature_envy\n"
          "Models: displays the machine learning models in the system and the classes they are used on\n"
          "Retrain: retrains trains all the models again using new train/test splits\n"
          "Quit: Quit exits the system. All models are deleted")
def compare(Data,model,modelName,className):
    print("Model:",modelName,"class",className)
    trainAcc=accuracy_score(Data[2], model.predict(Data[0]))
    testAcc=accuracy_score(Data[3], model.predict(Data[1]))
    trainF1=f1_score(Data[2], model.predict(Data[0]))
    testF1=f1_score(Data[3], model.predict(Data[1]))
    print(modelName,"Accuracy","F1-score",sep="\t\t")
    print("Training set",trainAcc,trainF1,sep="\t")
    print("Testing set",testAcc,testF1,sep="\t")

def run(Data,Model,modelname,indexClass,modelNum,classNames):
    print("model:",modelname)
    print("smell","Accuracy","F1-Score",sep="\t\t")
    num=0
    for index in indexClass:
        data=Data[index]
        model=Model[index][modelNum]
        testAcc=accuracy_score(data[3], model.predict(data[1]))
        testF1=f1_score(data[3], model.predict(data[1]))
        print(classNames[num],testAcc,testF1,sep="\t")
        num+=1
def setupModels():
    envyData, envyModels = mt.make_model('feature-envy.arff')
    # God class
    godData, godModels = mt.make_model('god-class.arff')
    # data class
    dataData, dataModels = mt.make_model('data-class.arff')
    # long method
    longData, longModels = mt.make_model('long-method.arff')
    Data = [envyData, godData, dataData, longData]
    Models = [envyModels, godModels, dataModels, longModels]
    return Data,Models
if __name__ == '__main__':
    numargs=len(sys.argv)
    if numargs>1:
        usage()
        sys.exit()
    print("Setting up models. Please wait...")
    # Feature envy
    Data,Models=setupModels()
    print("Setup complete")
    status=1
    classes=["feature_envy","god_class","data_class","long_method"]
    models=["decision_tree","random_forest","naive_bayes","svm_linear","svm_poly","svm_rbf","svm_sig"]
    while(status):
        A=input("Options are [Run | Compare | Models | Retrain | Usage | Quit]\n")
        if A.lower() == "quit" or A.lower()=='q':
            print("exiting")
            status=0
        elif A.lower()=="usage":
            usage()
        elif "compare" in A.lower():
            command=A.lower()
            parts = command.split(' ')
            if len(parts) != 3:
                print("error: compare takes 2 arguments.\nexample: compare decision_tree feature_envy")
            else:
                model = parts[1]
                _class = parts[2]
                if model in models and _class in classes:
                    modelNum=models.index(model)
                    classNum=classes.index(_class)
                    compare(Data[classNum],Models[classNum][modelNum],model,_class)

                else:
                    print("error: class or model name incorrect")
        elif "run" in A.lower():
            command = A.lower()
            parts = command.split(' ')
            if len(parts) < 3:
                print("error: run takes 2 or more arguments.\nrun decision_tree feature_envy god_class")
            else:
                length=len(parts)
                listClass=parts[2:length]
                model = parts[1]
                if (all(x in classes for x in listClass)) and model in models:
                    modelNum = models.index(model)
                    indexClasses=[]
                    for x in range(0,len(listClass)):
                        indexClasses.append(classes.index(listClass[x]))
                    run(Data,Models,model,indexClasses,modelNum,listClass)

                else:
                    print("error: class or model name incorrect")
        elif A.lower()=="models":
            print("Models are the following ",models)
            print("Classes are the following",classes)
        elif A.lower()=="retrain":
            print("Retraining models please wait...")
            Data, Models = setupModels()
            print("training complete")
        else:
            print(A,"Command unknown")


