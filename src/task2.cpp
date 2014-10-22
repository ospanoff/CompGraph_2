#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

Matrix<int> toGreyScale(BMP *in)
{
    Matrix<int> img(in->TellHeight(), in->TellWidth());
    for (int i = 0; i < in->TellHeight(); i++)
        for (int j = 0; j < in->TellWidth(); j++) {
            img(i, j) = static_cast<int>(0.299 * (*in)(j,i)->Red +
                                         0.587 * (*in)(j,i)->Green +
                                         0.114 * (*in)(j,i)->Blue);
        }

    return img;
}

// Sobel by unary map gives worse result at all
// so here hand-made sobel filter
pair<Matrix<int>, Matrix<int>> countSobel(BMP *in)
{
    Matrix<int> img(toGreyScale(in));
    Matrix<int> sobelX(toGreyScale(in));
    Matrix<int> sobelY(toGreyScale(in));
    
    int rows = static_cast<int>(sobelX.n_rows);
    int cols = static_cast<int>(sobelX.n_cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            sobelY(i, j) = ((i > 0) ? img(i - 1, j) : 0) - ((i < rows - 1) ? img(i + 1, j) : 0);
            sobelX(i, j) = ((j < cols - 1) ? img(i, j + 1) : 0) - ((j > 0) ? img(i, j - 1) : 0);
        }
    
    return make_pair(sobelX, sobelY);
}

pair<Matrix<float>, Matrix<float>> countModAndDirOfGrad(BMP *in)
{
    auto sobel = countSobel(in);
    Matrix<float> module(sobel.first.n_rows, sobel.first.n_cols);
    Matrix<float> direction(sobel.first.n_rows, sobel.first.n_cols);

    for (size_t i = 0; i < sobel.first.n_rows; i++)
        for (size_t j = 0; j < sobel.first.n_cols; j++) {
            auto x = sobel.first(i, j);
            auto y = sobel.second(i, j);
            module(i, j) = sqrt(x*x + y*y);
            direction(i, j) = atan2(y, x);
        }

    return make_pair(module, direction);
}

pair<float, float> phi(float x, float l)
{
    float a(0), b(0);
    if (x > 0) {
        a = cos(l * log(x)) * sqrt(x / cosh(M_PI * l));
        b = -sin(l * log(x)) * sqrt(x / cosh(M_PI * l));
    }
    return make_pair(a, b);
}

// Exatract features from dataset.
void ExtractFeatures(const TDataSet& data_set, TFeatures* features)
{
    const int blockSizeX(8);
    const int blockSizeY(8);
    const int dirSegSize(32);

    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        vector<float> one_image_features(blockSizeX * blockSizeY * dirSegSize, 0);

        BMP *image = data_set[image_idx].first;
        auto modDir = countModAndDirOfGrad(image);

        int rows = static_cast<int>(modDir.first.n_rows); // we use these ...
        int cols = static_cast<int>(modDir.first.n_cols); // ... not only one time
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                int blockIndx = static_cast<int>(i * blockSizeY / rows) * blockSizeY +
                                static_cast<int>(j * blockSizeX / cols);
                int angleIndx = static_cast<int>(((static_cast<int>(modDir.second(i, j)) + M_PI) /
                                                  (2 * M_PI)) * dirSegSize);
                int featIndx = blockIndx * dirSegSize + angleIndx;
                one_image_features[featIndx] += modDir.first(i, j);
        }

        int numOfBlocks(2); // normalizing blocks with norm of numOfBlocks blocks
        for (int i = 0; i < blockSizeX * blockSizeY; i += numOfBlocks) {
            float norm(0);
            for (int j = 0; j < dirSegSize * numOfBlocks; j++)
                norm += pow(one_image_features[i * dirSegSize + j], 2);

            norm = sqrt(norm);
            for (int j = 0; j < dirSegSize * numOfBlocks; j++)
                if (norm > 0) {
                    one_image_features[i * dirSegSize + j] /= norm;
                }
        }

        // nonlinear SVM
        const int n = 1;
        const float L = 0.5;
        vector<float> tmp;

        for (size_t i = 0; i < one_image_features.size(); i++) {
            for (int j = -n; j <= n; j++) {
                auto x = phi(one_image_features[i], j * L);
                tmp.push_back(x.first);
                tmp.push_back(x.second);
            }
        }

        features->push_back(make_pair(tmp /*one_image_features*/, data_set[image_idx].second));
        one_image_features.clear();
        tmp.clear();
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.1;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}