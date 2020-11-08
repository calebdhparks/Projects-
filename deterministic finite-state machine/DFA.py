#!/usr/bin/env python3
import sys 
#Caleb David Hammond Parks 
#create class to model a DFA
class DFA:
	global numberOfStates
	global finalStates
	global alphabet
	global transitionTable
	global startState
	global currentState
#Class to house Homomorphism	
class Homomorphism:
	global inAlphabet
	global outAlphabet
	global hTable

##This function Reads the comand line for a DFA file and populates an instance of the DFA class
#It takes in the index in argv that the file is located normally arg=2
def simulate (arg):
	dfa=DFA()
	DFA_file=open(sys.argv[arg],'r')
	DFA_lines=DFA_file.readlines()
	dfa.numberOfStates=int(DFA_lines[0].strip('\n').split(': ')[1])
	dfa.finalStates=DFA_lines[1].strip('\n').split(': ')[1].split(' ')
	alphabetString=DFA_lines[2].strip('\n').split(': ')[1]
	dfa.alphabet=list(alphabetString)
	dfa.transitionTable=[]
	for x in range (3,len(DFA_lines)):
		dfa.transitionTable.append(DFA_lines[x].strip('\n').split(' '))
	
	return dfa
#populates the Homomorphism class
def fillHomo (arg):
	homo=Homomorphism()
	file=open(sys.argv[arg],'r')
	lines=file.readlines()
	homo.inAlphabet=list(lines[0].strip('\n').split(': ')[1])
	homo.outAlphabet=list(lines[1].strip('\n').split(': ')[1])
	homo.hTable=[]
	for x in range (2,len(lines)):
		homo.hTable.append(lines[x].strip('\n'))
	return homo

##This function formats the ouput of a DFA to match the prefered format
#it prints to standard output
def writeDFA(dfa):
	print("Number of States: "+str(dfa.numberOfStates))
	print("Accepting States:",end=" ")
	for x in range (0,len(dfa.finalStates)):
		print(dfa.finalStates[x],end=" ")
	print("\nAlphabet: "+str(''.join(dfa.alphabet)))
	for x in range (0,len(dfa.transitionTable)):
		for y in range (0,len(dfa.transitionTable[x])):
			print(dfa.transitionTable[x][y],end=" ")
		print()
	return
##find what state a string leaves a DFA
def findState(dfa,string,fromState):
	state=fromState
	for x in range (0,len(string)):
		state=int(dfa.transitionTable[state][dfa.alphabet.index(list(string)[x])])
	return(state)
#

##./DFA.py -s <DFA_File> <Strings_File>
#Simulating a DFA
#Opens the strings file and tests each one to see if it puts the DFA in a final state
#It relies on the string function index(). alphabet.index() allows my to pass a character and its index in the alphabet table is returned which is also its index in the transistion table 
#prints either 'accept' or 'reject' to standard output
if sys.argv[1]=="-s":
	dfa=simulate(2)
	dfa.startState=0
	dfa.currentState=0
	string_file=open(sys.argv[3],'r')
	strings=string_file.readlines()
	for line in strings:
		if line.strip('\n') !="":
			dfa.currentState=0
			chars=list(line.strip('\n'))
			for char in chars:
				dfa.currentState=dfa.transitionTable[int(dfa.currentState)][dfa.alphabet.index(char)]
			if str(dfa.currentState) in dfa.finalStates:
				print("accept")
			else:
				print("reject")
		else:
			dfa.currentState=0
			if str(dfa.currentState) in dfa.finalStates:
				print("accept")
			else:
				print("reject")
##./DFA.py -t <String_File>
#Text search
#Can be used to build more than one DFA if multiple strings are in String_File
#Breaks the transition table into three cases
#(1) The character you read is the correct character 
#(2) The character you read is not in the set of characters the DFA is looking for. EX reading a c when the string is abbba
#(3) The character you read is not the correct character but does appear in the string 
	#Here have to find the where to transistion by matching strings of increasing length until we find a match. then we take the max of all the matches 
elif sys.argv[1]=='-t':
	dfa=DFA()
	stringFile=open(sys.argv[2],'r')
	strings=stringFile.readlines()
	for line in strings:
		char=list(line.strip('\n'))
		thisLine=line.strip('\n')
		dfa.numberOfStates=len(char)+1
		dfa.finalStates=[len(char)]
		dfa.alphabet=list("abcdefghijklmnopqrstuvwxyz")
		temp=[]
		dfa.transitionTable=[]
		setChar=sorted(set(char))
		temp=[]
		for x in range(0, len(char)):
			temp=[]
			for letter in range (0,len(dfa.alphabet)):
				newLine=thisLine[0:x]+str(dfa.alphabet[letter])
				#(1)
				if newLine == thisLine[0:x+1]:
					temp.append(x+1)
				#(2)
				elif dfa.alphabet[letter] not in setChar:
					temp.append(0)
				#(3)
				else:	
					endPrefix=1
					startSuffix=x
					states=[0]
					while endPrefix<=x:
						Prefix=newLine[0:endPrefix]
						Suffix=newLine[startSuffix:x+1]
						if Prefix==Suffix:
							states.append(endPrefix)
						endPrefix=endPrefix+1
						startSuffix=startSuffix-1
					temp.append(max(states))
									
			dfa.transitionTable.append(temp)
		temp=[]
		for y in range (0,len(dfa.alphabet)):
				temp.append(len(char))
		dfa.transitionTable.append(temp)
		writeDFA(dfa)
#./DFA -c <DFA_File>
#Complement
#	or
#./DFA -c <DFA_File1> <DFA_File2>
#intersection
#Closure properties: complement and intersection
#Complement just flips the accepting states 
#Intersection creates a stateMap table that holds all of the possible combination of new states. using the .index() function gives me my mapping from two states to one
#EX. state 0 ind DFA1 and state 0 in DFA2 is inserted into stateMap as '0 0' and doing stateMap.index('0 0') returns 0
#0 1 -> 1
#0 2 -> 2 and so on
#It assumes the two DFAs have the same alphabet 
#writes a DFA to standard output 
elif sys.argv[1] =='-c' and len(sys.argv)==3:
	dfa=simulate(2)
	newFinal=[]
	for i in range(0,dfa.numberOfStates):
		if str(i) not in dfa.finalStates:
			newFinal.append(i)
	dfa.finalStates=newFinal
	writeDFA(dfa)
elif sys.argv[1] =='-c' and len(sys.argv)==4:
	dfa1=simulate(2)
	dfa2=simulate(3)	
	numStates=int(dfa1.numberOfStates)*int(dfa2.numberOfStates)
	stateMap=[]
	fState=[]
	for x in range(0,dfa1.numberOfStates):
		for y in range (0,dfa2.numberOfStates):
			stateMap.append(str(x)+" "+str(y))
			if str(x) in dfa1.finalStates and str(y) in dfa2.finalStates:
				fState.append(stateMap.index(str(x)+" "+str(y)))
	transition=[]
	temp=[]
	for x in range (0,len(stateMap)):
		temp=[]
		for y in range (0,len(dfa1.alphabet)):
			char=stateMap[x].split(" ")
			firstState=str(dfa1.transitionTable[int(char[0])][y])
			secondState=str(dfa2.transitionTable[int(char[1])][y])
			lookup=firstState+" "+secondState
			temp.append(stateMap.index(lookup))
		transition.append(temp)
	newDfa=DFA()
	newDfa.startState=0
	newDfa.currentState=0
	newDfa.numberOfStates=numStates
	newDfa.finalStates=fState
	newDfa.alphabet=dfa1.alphabet
	newDfa.transitionTable=transition
	writeDFA(newDfa)
## ./DFA.py -i <DFA_file> <Homo_file>
#inverse
#Finds what state each homomorphism puts the DFA in
elif sys.argv[1] == '-i':
	dfa=simulate(2)
	homo=fillHomo(3)
	newDfa=DFA()
	newDfa.startState=0
	newDfa.currentState=0
	newDfa.numberOfStates=dfa.numberOfStates
	newDfa.finalStates=dfa.finalStates
	newDfa.alphabet=homo.inAlphabet
	newDfa.transitionTable=[]
	for x in range(0,newDfa.numberOfStates):
		temp=[]
		for y in range(0,len(newDfa.alphabet)):
			temp.append(findState(dfa,homo.hTable[y],x))
		newDfa.transitionTable.append(temp)
	writeDFA(newDfa)
		

			
	

