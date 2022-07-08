"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder)]
    size = tuple(size)
    img = []
    label = []
    for im in images_files:
        imgTemp = cv2.imread(os.path.join(folder, im))
        #imgTemp = cv2.cvtColor(imgTemp, cv2.COLOR_BGR2GRAY)
        imgTemp = cv2.resize(imgTemp, size)
        imgTemp = imgTemp.flatten()
        img.append(imgTemp)
        labelTemp = im.split('.')[0][-2:] 
        #print(labelTemp)
        labelTemp = int(labelTemp)
        label.append(labelTemp)
    #img = np.vstack(img)
    #label = np.vstack(label)
    return (np.array(img), np.array(label))
        

    #raise NotImplementedError


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """

    #raise NotImplementedError
    n = len(y)
    #print(n)
    trained = int(n * p)
    idx = np.random.permutation(n)
    trainedIDX = idx[:trained].astype(int)
    testIDX = idx[trained:].astype(int)
    #https://stackoverflow.com/questions/50997928/typeerror-only-integer-scalar-arrays-can-be-converted-to-a-scalar-index-with-1d
    #print(np.array(y)[trainedIDX.astype(int)])
    #print( y[testIDX])
    #print(testIDX)
    return np.array(X)[trainedIDX, :],np.array(y)[trainedIDX], np.array(X)[testIDX, :], np.array(y)[testIDX]


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    #raise NotImplementedError
    output = np.mean(x, axis=0, dtype=float)
    return output

def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    #raise NotImplementedError
    phi = X - get_mean_face(X)
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(phi.T, phi))
    return eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]], eigenvalues[np.argsort(eigenvalues)[::-1][:k]]

class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        #raise NotImplementedError
        #for j = 1...M do
        for i in range(self.num_iterations):
            #Renormalize the weights so they sum up to 1
            self.weights /= np.sum(self.weights)
            #Instantiate the weak classifier h with the training data and labels. 
            h = WeakClassifier(X = self.Xtrain, y = self.ytrain, weights = self.weights)
            # Train the classifier h ;
            h.train()
            self.weakClassifiers.append(h)
            #Get predictions h(x) for all training examples x ∈ Xtr ai n
            pred = h.predict(np.transpose(self.Xtrain))
            #Find ϵj =Piwi for weights where h(xi) ̸= yi
            _erroSum = []
            _erroSumI = []
            for i in range(len(pred)):
                if pred[i]!=self.ytrain[i]:
                    _erroSum.append(self.weights[i])
                    _erroSumI.append(i)
            erroSum = np.sum(_erroSum)
            #Calculate α
            alpha = 0.5 * np.log((1.0 - erroSum)/erroSum)
            self.alphas.append(alpha)
            if erroSum > self.eps:
                #Update the weights
                self.weights[_erroSumI] = self.weights[_erroSumI] * np.exp(-alpha * pred[_erroSumI] * self.ytrain[_erroSumI])
            else:
                break
                
                    

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        #raise NotImplementedError
        pred = self.predict(self.Xtrain)
        correct = 0
        incorrect = 0
        for pY, tY in zip(pred, self.ytrain):
            if pY == tY:
                correct+=1
            else:
                incorrect+=1
        return correct, incorrect
        

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        #raise NotImplementedError
        pred = []
        for i in range(len(self.alphas)):
            pr = self.weakClassifiers[i].predict(np.transpose(X))
            pr = np.array(pr)*self.alphas[i]
            pred.append(pr)
        pred = np.sum(pred, axis = 0)
        #print(pred)
        return np.sign(pred)
          
                     


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        #raise NotImplementedError
        yLoc, xLoc = self.position
        height, width = self.size
        #initiation
        img = np.zeros(shape)
        #white
        img[yLoc:yLoc+int(height/2), xLoc:xLoc+width] = 255
        #gray
        img[yLoc+int(height/2):yLoc+height, xLoc:xLoc+width] = 126
        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        #raise NotImplementedError
        yLoc, xLoc = self.position
        height, width = self.size
        #initiation
        img = np.zeros(shape)
        #white
        img[yLoc:yLoc+int(height), xLoc:xLoc+int(width/2)] = 255
        #gray
        img[yLoc:yLoc+height, xLoc+int(width/2):xLoc+int(width)] = 126
        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        #raise NotImplementedError
        yLoc, xLoc = self.position
        height, width = self.size
        hDif = int(height/3.)
        #print(hDif)
        #print(yLoc+int(hDif))
        #print(yLoc+int(2*hDif))
        #print(xLoc)
        #print(xLoc+width)
        
        #initiation
        img = np.zeros(shape)
        #white
        img[yLoc:yLoc+int(hDif), xLoc:xLoc+width] = 255
        #gray
        img[yLoc+int(hDif):yLoc+int(2*hDif), xLoc:xLoc+width] = 126
        #white
        img[yLoc+int(2*hDif):yLoc+height, xLoc:xLoc+width] = 255
        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        #raise NotImplementedError
        yLoc, xLoc = self.position
        height, width = self.size
        wDif = int(width/3.)
        #initiation
        img = np.zeros(shape)
        #white
        img[yLoc:yLoc+height, xLoc:xLoc+int(wDif)] = 255
        #gray
        img[yLoc:yLoc+height, xLoc+int(wDif):xLoc+int(2 * wDif)] = 126
        #white
        img[yLoc:yLoc+height, xLoc+int(2 * wDif):xLoc+width] = 255
        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        #raise NotImplementedError
        yLoc, xLoc = self.position
        height, width = self.size
        difH, difW = int(height/2),int(width/2)
        #initiation
        img = np.zeros(shape)
        
        img[yLoc:yLoc+difH, xLoc:xLoc+difW] = 126
        img[yLoc:yLoc+difH, xLoc+difW:xLoc+width] = 255
        img[yLoc+difH:yLoc+height, xLoc:xLoc+difW] = 255
        img[yLoc+difH:yLoc+height, xLoc+difW:xLoc+width] = 126
        return img
       

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        #raise NotImplementedError
        score = 0
        y, x = self.position
        h, w = self.size
        ii = ii.astype(np.float64)
        y = 1 if y < 1 else y
        x = 1 if x < 1 else x 
        y-=1
        x-=1
        if self.feat_type == (2,1):
            #white
            W = ii[y,x] - ii[y+(h//2),x] - ii[y,x+w] + ii[y+(h//2), x+w]
            #print(W)
            #gray
            G = ii[y+(h//2),x] - ii[y+h,x] - ii[y+(h//2),x+w] + ii[y+h,x+w]
            #print(G)
            score = W-G
        elif self.feat_type == (1,2):
            #white
            W = ii[y,x] - ii[y+h,x] - ii[y,x+(w//2)] + ii[y+h, x+(w//2)]
            #gray
            G = ii[y,x+(w//2)] - ii[y+h,x+(w//2)] - ii[y,x+w] + ii[y+h,x+w]
            score = W-G
        elif self.feat_type == (3,1):
            #white
            W = ii[y,x] - ii[y+(h//3),x] - ii[y,x+w] + ii[y+(h//3), x+w]
            #gray
            G = ii[y+(h//3),x] - ii[y+(2*h//3),x] - ii[y+(h//3),x+w] + ii[y+(2*h//3),x+w]
            #white
            W1 = ii[y+(2*h//3),x] - ii[y+(h),x] - ii[y+(2*h//3),x+w] + ii[y+(h), x+w]
            score = W-G+W1
        elif self.feat_type == (1,3):
            #white
            W = ii[y,x] - ii[y+h,x] - ii[y,x+(w//3)] + ii[y+h, x+(w//3)]
            #gray
            G = ii[y,x+(w//3)] - ii[y+h,x+(w//3)] - ii[y,x+(w*2//3)] + ii[y+h,x+(w*2//3)]
            #white
            W1 = ii[y,x+(w*2//3)] - ii[y+h,x+(w*2//3)] - ii[y,x+(w)] + ii[y+h, x+(w)]
            score = W-G+W1
        else:
            #gray
            G1 = ii[y,x] - ii[y+(h//2),x] - ii[y,x+(w//2)] + ii[y+(h//2), x+(w//2)]
            #white
            W1 = ii[y,x+(w//2)] - ii[y+(h//2),x+(w//2)] - ii[y,x+w] + ii[y+(h//2),x+w]
            #white
            W2 = ii[y+(h//2),x] - ii[y+(h),x] - ii[y+(h//2),x+(w//2)] + ii[y+(h),x+(w//2)]
            #gray
            G2 = ii[y+(h//2),x+(w//2)] - ii[y+(h),x+(w//2)] - ii[y+(h//2),x+(w)] + ii[y+(h), x+(w)]
            score = W1-G1+W2-G2
            #print(G1)
            #print(W1)
            #print(W2)
            #print(G2)
        return score

def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    #raise NotImplementedError
    ii = []
    for i in images:
        ii.append(np.cumsum(np.cumsum(i, axis = 0), axis = 1))
    return ii

class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
        self.threshold = 1.0

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_train(self):
        """ This function initializes self.scores, self.weights

        Args:
            None

        Returns:
            None
        """
    
        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        if not self.integralImages or not self.haarFeatures:
            print("No images provided. run convertImagesToIntegralImages() first")
            print("       Or no features provided. run creatHaarFeatures() first")
            return

        self.scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            self.scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        self.weights = np.hstack((weights_pos, weights_neg))

    def train(self, num_classifiers):
        """ Initialize and train Viola Jones face detector

        The function should modify self.weights, self.classifiers, self.alphas, and self.threshold

        Args:
            None

        Returns:
            None
        """
        self.init_train()
        print(" -- select classifiers --")
        weights = self.weights
        for i in range(num_classifiers):
            # TODO: Complete the Viola Jones algorithm
            #raise NotImplementedError
            weightsSum = np.sum(weights)
            weights = weights/weightsSum
            clf = VJ_Classifier(self.scores, self.labels, weights)
            clf.train()
            self.classifiers.append(clf)
            err = clf.error
            B = err/(1.-err)
            alpha = np.log(1./B)
            pred = clf.predict(np.transpose(self.scores))
            e = np.zeros(np.shape(pred))
            self.alphas.append(alpha)
            for i in range(len(pred)):
                if pred[i] != self.labels[i]:
                    e[i] = 1
                else:
                    e[i] = -1
            for i in range(len(weights)):
                weights[i] = weights[i] * pow(B, 1-e[i])


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.
        for clf in self.classifiers:
        # Obtain the Haar feature id from clf.feature
            featureID = clf.feature
        # Use this id to select the respective feature object from
        # self.haarFeatures
            f = self.haarFeatures[featureID]
        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'
            for x in range(len(ii)):
                scores[x, featureID] = f.evaluate(ii[x])
        
        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        
        for x in scores:
            # TODO
            #raise NotImplementedError
            H = []
            for i in range(len(self.alphas)):
                H.append(self.alphas[i] * self.classifiers[i].predict(x))
            if np.sum(H) < 0.5*np.sum(self.alphas):
                result.append(-1)
            else:
                result.append(1)
            
        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        #raise NotImplementedError
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        collections = []
        locs = []
        for j in range(0, image.shape[0]-24):
            for i in range(0, image.shape[1]-24):
                collections.append(img[j:j+24, i:i+24])
                locs.append((j,i))  
        pred = self.predict(collections)
        
        J = 0
        I = 0
        count = 0
        for n in range(len(locs)):
            if pred[n] == 1:
                J+=locs[n][0]
                I+=locs[n][1]
                count += 1
        J/=count
        I/=count
        #print(J,I)
        J = int(J)
        I = int(I)
        cv2.rectangle(image, (I, J), (I+24,J+24), (255,0,0), 1)
        if filename is not None:
            cv2.imwrite('output/'+filename+'.png', image)
                        
        
                

class CascadeClassifier:
    """Viola Jones Cascade Classifier Face Detection Method

    Lesson: 8C-L2, Boosting and face detection

    Args:
        f_max (float): maximum acceptable false positive rate per layer
        d_min (float): minimum acceptable detection rate per layer
        f_target (float): overall target false positive rate
        pos (list): List of positive images.
        neg (list): List of negative images.

    Attributes:
        f_target: overall false positive rate
        classifiers (list): Adaboost classifiers
        train_pos (list of numpy arrays):  
        train_neg (list of numpy arrays): 

    """
    def __init__(self, pos, neg, f_max_rate=0.30, d_min_rate=0.70, f_target = 0.07):
        
        train_percentage = 0.85

        pos_indices = np.random.permutation(len(pos)).tolist()
        neg_indices = np.random.permutation(len(neg)).tolist()

        train_pos_num = int(train_percentage * len(pos))
        train_neg_num = int(train_percentage * len(neg))

        pos_train_indices = pos_indices[:train_pos_num]
        pos_validate_indices = pos_indices[train_pos_num:]

        neg_train_indices = neg_indices[:train_neg_num]
        neg_validate_indices = neg_indices[train_neg_num:]

        self.train_pos = [pos[i] for i in pos_train_indices]
        self.train_neg = [neg[i] for i in neg_train_indices]

        self.validate_pos = [pos[i] for i in pos_validate_indices]
        self.validate_neg = [neg[i] for i in neg_validate_indices]

        self.f_max_rate = f_max_rate
        self.d_min_rate = d_min_rate
        self.f_target = f_target
        self.classifiers = []

    def predict(self, classifiers, img):
        """Predict face in a single image given a list of cascaded classifiers

        Args:
            classifiers (list of element type ViolaJones): list of ViolaJones classifiers to predict 
                where index i is the i'th consecutive ViolaJones classifier
            img (numpy.array): Input image

        Returns:
            Return 1 (face detected) or -1 (no face detected) 
        """

        # TODO
        #raise NotImplementedError

    def evaluate_classifiers(self, pos, neg, classifiers):
        """ 
        Given a set of classifiers and positive and negative set
        return false positive rate and detection rate 

        Args:
            pos (list): Input image.
            neg (list): Output image file name.
            classifiers (list):  

        Returns:
            f (float): false positive rate
            d (float): detection rate
            false_positives (list): list of false positive images
        """

        # TODO
        #raise NotImplementedError

    def train(self):
        """ 
        Trains a cascaded face detector

        Sets self.classifiers (list): List of ViolaJones classifiers where index i is the i'th consecutive ViolaJones classifier

        Args:
            None

        Returns:
            None
             
        """
        # TODO
        #raise NotImplementedError


    def faceDetection(self, image, filename="ps6-5-b-1.jpg"):
        """Scans for faces in a given image using the Cascaded Classifier.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        #raise NotImplementedError