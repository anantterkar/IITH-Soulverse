from collections import Counter
from deepface import DeepFace
#this function is for counting number of matching values in two vector embeddings
def matchingcnt(x, y, img1, img2, mmi):
  cnt=0
  for i in range(512):
    if(x[i] == y[i]):
      cnt=cnt+1
    else:
      mmi.append(i)
  p = (cnt/512)*100
  print("Image {} and Image {}".format(img1, img2))
  print("Total Matching Values = ", cnt)
  print("Percentage of Matching = {}%".format(p))
  print("=====================================")

#for counting number of matching values in filtered vector embeddings
def matchingcnt_filtered(x,y,mcnt):
  cnt=0
  for i in range(mcnt):
    if(x[i]==y[i]):
      cnt=cnt+1
  p = (cnt/mcnt)*100
  print("Total Matching Values = ", cnt)
  print("After filtering {}%".format(p))
  print("=====================================")
  return p

#hashing of indices
def mismatch_index(mmi):
  frq = Counter(mmi)
  return list(frq.keys())

#picks anchor vector by selecting the closest vector to all other vectors
def anchorpick(matrix):
  num_vectors  = matrix.shape[1]
  distance_sums = np.zeros(num_vectors)
  for i in range(num_vectors):
    for j in range(num_vectors):
      if i!=j:
        distance_sums[i] += np.linalg.norm(matrix[:, i] - matrix[:, j])

  anchor_index = np.argmin(distance_sums)
  return matrix[:,anchor_index], anchor_index

def down_mapping(arr, vlen):
  for i in range(vlen):
    if(arr[i]==-2):
      arr[i]=1
    elif(arr[i]==-1):
      arr[i]=2
    elif(arr[i]==0):
      arr[i]=3
    elif(arr[i]==1):
      arr[i]=4
    elif(arr[i]==2):
      arr[i]=5
  v128 = np.zeros(128)
  for i in range(0,vlen-1,4):
    vinx = int(i/4)
    v128[vinx] = arr[i]*1000 + arr[i+1]*100 + arr[i+2]*10 + arr[i+3]
  return v128

#function for generating embeddings using Facenet512 
def getembeddings(imgpaths):
  array_embeddings = []
  for i in range(len(imgpaths)):  #loops over all images of the person
    embedding = DeepFace.represent(img_path = imgpaths[i],
                                 model_name = "Facenet512", detector_backend='retinaface')
    embarr = np.array(embedding[0]["embedding"])
    embarr = np.array(embarr).round(0)
    array_embeddings.append(embarr)
  embmatrix = np.column_stack(array_embeddings)
  print(embmatrix.size)
  return embmatrix

#primary function
def mismatch(matrix):
  mmi = []
  anchor, ainx = anchorpick(matrix)
  for i in range(0,4):
    for j in range((i+1), 5):
      if(i!=ainx and j!=ainx):
        x = matrix[:,i]
        y = matrix[:,j]
      #print(x)
      #print(y)
        matchingcnt(x,y,i+1,j+1,mmi)
  keys_list = mismatch_index(mmi)
  mask = np.ones(512, dtype=bool)
  mask[keys_list] = False
# Apply the mask to get the filtered array
  filtered_anchor = anchor[mask]
  filtered_embarr2 = matrix[:,1][mask]
  filtered_embarr3 = matrix[:,2][mask]
  filtered_embarr4 = matrix[:,3][mask]
  filtered_embarr5 = matrix[:,4][mask]
  mcnt = 512-len(keys_list)
  return matchingcnt_filtered(filtered_anchor, filtered_embarr2, mcnt), mcnt, filtered_anchor
