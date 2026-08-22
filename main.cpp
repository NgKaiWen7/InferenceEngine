#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

cv::Mat cropText(const cv::Mat &src, const std::array<cv::Point2f, 4> &box);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./ocr_engine <image>\n";
        return 1;
    }
    const char *filename = argv[1];

    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty())
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Log");
        std::cerr << "Failed to load image: " << filename << "\n";
        return 1;
    }
    cv::Mat floatImage;

    int imgH = image.rows;
    int imgW = image.cols;
    float ratio = static_cast<float>(imgW) / imgH;
    int max_value = std::max(imgH, imgW);
    int min_value = std::min(imgH, imgW);
    int new_h = imgH;
    int new_w = imgW;
    if (max_value > 2000)
    {
        if (imgH > imgW)
        {
            ratio = 2000.0f / imgH;
        }
        else
        {
            ratio = 2000.0f / imgW;
        }
        new_h = static_cast<int>(imgH * ratio);
        new_w = static_cast<int>(imgW * ratio);
    }

    if (min_value < 30)
    {
        if (imgH < imgW)
        {
            ratio = 30.0f / imgH;
        }
        else
        {
            ratio = 30.0f / imgW;
        }
        new_h = static_cast<int>(imgH * ratio);
        new_w = static_cast<int>(imgW * ratio);
    }
    new_h = static_cast<int>(std::round(new_h / 32.0) * 32);
    new_w = static_cast<int>(std::round(new_w / 32.0) * 32);
    // cv::resize(image, image, cv::Size(new_w, new_h));
    image.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    int srcH = floatImage.rows;
    int srcW = floatImage.cols;

    std::vector<cv::Mat> channels;
    cv::split(floatImage, channels);
    std::vector<float> inputTensorValues(3 * srcH * srcW);

    std::memcpy(inputTensorValues.data(), channels[0].ptr<float>(), srcH * srcW * sizeof(float));
    std::memcpy(inputTensorValues.data() + srcH * srcW, channels[1].ptr<float>(), srcH * srcW * sizeof(float));
    std::memcpy(inputTensorValues.data() + 2 * srcH * srcW, channels[2].ptr<float>(), srcH * srcW * sizeof(float));

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Log");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);                                                // Set CPU threads for intra-op parallelization
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // Enable optimizations
    const char *det_model = "models/ch_PP-OCRv4_det_infer.onnx";
    Ort::Session detection_session(env, det_model, session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::MemoryInfo memoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    auto inputName = detection_session.GetInputNameAllocated(0, allocator);
    auto outputName = detection_session.GetOutputNameAllocated(0, allocator);

    const char *input_names[] = {inputName.get()};
    const char *output_names[] = {outputName.get()};

    Ort::TypeInfo type_info = detection_session.GetInputTypeInfo(0);
    std::vector<int64_t> inputShape = {1, 3, srcH, srcW};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputShape.data(),
        inputShape.size());

    auto outputTensors = detection_session.Run(Ort::RunOptions{nullptr}, input_names, &inputTensor, 1, output_names, 1);
    auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto outputShape = outputInfo.GetShape();
    int outH = static_cast<int>(outputShape[2]);
    int outW = static_cast<int>(outputShape[3]);
    float *data = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat pred(outH, outW, CV_32F, data);
    cv::Mat det_visual;
    pred.convertTo(det_visual, CV_8U, 255.0);
    cv::imwrite("det_output.png", det_visual);

    // DBPostProcess
    float thresh = 0.3f;
    cv::Mat bitmap;
    cv::threshold(pred, bitmap, thresh, 255, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8U);
    cv::Mat dilationKernel = cv::Mat::ones(2, 2, CV_8U);
    cv::dilate(bitmap, bitmap, dilationKernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap * 255, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int maxCandidates = 1000;
    int numContours = std::min(static_cast<int>(contours.size()), maxCandidates);

    std::vector<std::array<cv::Point2f, 4>> dtBoxes;
    std::vector<float> scoreVector;
    std::vector<cv::Mat> cropped_images;
    float boxThresh = 0.7f;
    float minSize = 3.0f;
    float unclip_ratio = 2.0;
    for (int i = 0; i < numContours; i++)
    {
        const auto &contour = contours[i];
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float shortSide = std::min(rect.size.width, rect.size.height);
        if (shortSide < minSize)
        {
            continue;
        }
        cv::Point2f points[4];
        rect.points(points);
        std::vector<cv::Point2f> boxPoints(points, points + 4);
        std::sort(boxPoints.begin(), boxPoints.end(),
                  [](const cv::Point2f &a, const cv::Point2f &b)
                  { return a.x < b.x; });

        int index1, index2, index3, index4;
        if (boxPoints[1].y > boxPoints[0].y)
        {
            index1 = 0;
            index4 = 1;
        }
        else
        {
            index1 = 1;
            index4 = 0;
        }
        if (boxPoints[3].y > boxPoints[2].y)
        {
            index2 = 2;
            index3 = 3;
        }
        else
        {
            index2 = 3;
            index3 = 2;
        }
        std::vector<cv::Point2f> box = {
            boxPoints[index1],
            boxPoints[index2],
            boxPoints[index3],
            boxPoints[index4]};
        cv::Mat mask = cv::Mat::zeros(pred.size(), CV_8U);
        std::vector<cv::Point> polygon;
        for (const auto &p : box)
            polygon.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255));
        float score = static_cast<float>(cv::mean(pred, mask)[0]);
        if (score < boxThresh)
        {
            continue;
        }
        float area = rect.size.width * rect.size.height;
        float perimeter = 2.0f * (rect.size.width + rect.size.height);

        float distance = area * unclip_ratio / perimeter;

        rect.size.width += 2.0f * distance;
        rect.size.height += 2.0f * distance;
        cv::Point2f expandedPoints[4];
        rect.points(expandedPoints);
        std::vector<cv::Point2f> expandedBoxPoints(
            expandedPoints,
            expandedPoints + 4);
        std::sort(expandedBoxPoints.begin(), expandedBoxPoints.end(),
                  [](const cv::Point2f &a, const cv::Point2f &b)
                  {
                      return a.x < b.x;
                  });

        cv::Point2f tl, tr, br, bl;

        if (expandedBoxPoints[0].y < expandedBoxPoints[1].y)
        {
            tl = expandedBoxPoints[0];
            bl = expandedBoxPoints[1];
        }
        else
        {
            tl = expandedBoxPoints[1];
            bl = expandedBoxPoints[0];
        }

        if (expandedBoxPoints[2].y < expandedBoxPoints[3].y)
        {
            tr = expandedBoxPoints[2];
            br = expandedBoxPoints[3];
        }
        else
        {
            tr = expandedBoxPoints[3];
            br = expandedBoxPoints[2];
        }

        std::array<cv::Point2f, 4> finalBox = {
            tl, tr, br, bl};
        cv::Mat cropped_image = cropText(image, finalBox);
        dtBoxes.push_back(finalBox);
        scoreVector.push_back(score);
        cropped_images.push_back(cropped_image);
    }

    // Visualize the output for debugging purpose
    cv::Mat visual = image.clone();
    for (const auto &box : dtBoxes)
    {
        std::vector<cv::Point> points;
        for (const auto &p : box)
        {
            points.emplace_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        }

        cv::polylines(visual, points, true, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("det_boxes.png", visual);

    float max_wh_ratio = srcW / srcH;
    for (const cv::Mat &images : cropped_images)
    {
        int current_col = images.cols;
        int current_row = images.rows;
        float ratio = current_col / current_row;
        max_wh_ratio = std::max(max_wh_ratio, ratio);
    }

    const char *rec_model = "models/ch_PP-OCRv4_rec_infer.onnx";
    Ort::Session recognition_session(env, rec_model, session_options);

    auto reg_inputName = recognition_session.GetInputNameAllocated(0, allocator);
    auto reg_outputName = recognition_session.GetOutputNameAllocated(0, allocator);

    const char *reg_input_names[] = {reg_inputName.get()};
    const char *reg_output_names[] = {reg_outputName.get()};
    Ort::TypeInfo reg_type_info = recognition_session.GetInputTypeInfo(0);
    Ort::ConstTensorTypeAndShapeInfo reg_type_tensor_info = reg_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> reg_input_shape = reg_type_tensor_info.GetShape();
    auto metadata = recognition_session.GetModelMetadata();
    auto character_map =
        metadata.LookupCustomMetadataMapAllocated(
            "character",
            allocator);
    std::string characters = character_map.get();
    std::vector<std::string> character_list;
    std::stringstream ss(characters);
    std::string line;
    while (std::getline(ss, line))
    {
        character_list.push_back(line);
    }
    character_list.push_back(" ");
    character_list.insert(character_list.begin(), "blank");
    int counter = 0;
    for (cv::Mat &temp_img : cropped_images)
    {
        int current_col = temp_img.cols;
        int current_row = temp_img.rows;

        float ratio = static_cast<float>(current_col) / current_row;

        int new_width = static_cast<int>(current_row * max_wh_ratio);

        int resized_w = std::min(
            new_width,
            static_cast<int>(std::ceil(current_row * ratio)));

        cv::Mat resized_image;

        cv::resize(
            temp_img,
            resized_image,
            cv::Size(resized_w, current_row));

        cv::Mat floatImage;

        cv::imwrite("counter_" + std::to_string(counter) + ".png", resized_image);
        counter++;
        resized_image.convertTo(
            floatImage,
            CV_32FC3,
            1.0 / 255.0);

        cv::Mat padded = cv::Mat::zeros(
            current_row,
            current_col,
            CV_32FC3);

        floatImage.copyTo(
            padded(cv::Rect(
                0,
                0,
                resized_w,
                current_row)));

        padded = padded - 0.5f;
        padded = padded / 0.5f;

        std::vector<cv::Mat> channels;
        cv::split(padded, channels);
        std::cout << "padded: "
                  << padded.cols << " x "
                  << padded.rows << std::endl;

        std::cout << "tensor shape: [1, 3, "
                  << padded.rows << ", "
                  << padded.cols << "]" << std::endl;

        int planeSize = padded.cols * padded.rows;
        std::vector<float> reg_inputTensorValues(3 * planeSize);
        std::vector<int64_t> reg_inputShape = {1, 3, padded.rows, padded.cols};

        std::memcpy(reg_inputTensorValues.data(),
                    channels[0].ptr<float>(),
                    planeSize * sizeof(float));

        std::memcpy(reg_inputTensorValues.data() + planeSize,
                    channels[1].ptr<float>(),
                    planeSize * sizeof(float));

        std::memcpy(reg_inputTensorValues.data() + 2 * planeSize,
                    channels[2].ptr<float>(),
                    planeSize * sizeof(float));
        Ort::Value reg_inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            reg_inputTensorValues.data(),
            reg_inputTensorValues.size(),
            reg_inputShape.data(),
            reg_inputShape.size());
        auto outputTensors = recognition_session.Run(
            Ort::RunOptions{nullptr}, reg_input_names, &reg_inputTensor, 1, reg_output_names, 1);
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        auto outputShape = outputInfo.GetShape();
        // int64_t batch = outputShape[0];
        int64_t T = outputShape[1];
        int64_t numClasses = outputShape[2];
        std::cout << "T = " << T << std::endl;
        std::cout << "numClasses = " << numClasses << std::endl;
        std::cout << "character_list.size() = "
                  << character_list.size() << std::endl;
        const float *data = outputTensors[0].GetTensorData<float>();
        std::vector<int> indices;
        std::vector<float> scores;

        indices.reserve(T);
        scores.reserve(T);
        for (int64_t t = 0; t < T; t++)
        {
            // 1. Get 6625 probabilities for this timestep
            const float *row = data + t * numClasses;

            // 2. Argmax over classes
            int maxIndex = 0;
            float maxScore = row[0];

            for (int64_t c = 1; c < numClasses; c++)
            {
                if (row[c] > maxScore)
                {
                    maxScore = row[c];
                    maxIndex = static_cast<int>(c);
                }
            }
            std::cout << maxIndex << " ";
            if (!indices.empty() &&
                maxIndex == indices.back())
            {
                continue;
            }

            if (maxIndex != 0 && maxIndex < character_list.size())
            {
                indices.push_back(maxIndex);
                scores.push_back(maxScore);
            }
        }
        std::cout << std::endl;
        for (int i : indices)
        {
            std::cout << character_list[i];
        }
        std::cout << std::endl;
    }
    return 0;
}

cv::Mat cropText(const cv::Mat &src, const std::array<cv::Point2f, 4> &box)
{
    float width1 = cv::norm(box[1] - box[0]);
    float width2 = cv::norm(box[2] - box[3]);
    float height1 = cv::norm(box[3] - box[0]);
    float height2 = cv::norm(box[2] - box[1]);

    int width = static_cast<int>(std::round(std::max(width1, width2)));
    int height = static_cast<int>(std::round(std::max(height1, height2)));
    if (width <= 0 || height <= 0)
        return {};
    std::vector<cv::Point2f> srcPts = {
        box[0], box[1], box[2], box[3]};

    std::vector<cv::Point2f> dstPts = {
        {0, 0},
        {static_cast<float>(width - 1), 0},
        {static_cast<float>(width - 1), static_cast<float>(height - 1)},
        {0, static_cast<float>(height - 1)}};

    cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);

    cv::Mat crop;
    cv::warpPerspective(src, crop, M, cv::Size(width, height));

    return crop;
}

float polygonArea(const std::array<cv::Point2f, 4> &p)
{
    float area = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        area += p[i].x * p[j].y - p[j].x * p[i].y;
    }

    return std::abs(area) * 0.5f;
}

float polygonPerimeter(const std::array<cv::Point2f, 4> &p)
{
    float perimeter = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        perimeter += cv::norm(p[j] - p[i]);
    }

    return perimeter;
}