#ifndef MNIST_LIBTORCH_CPP_MODEL_H
#define MNIST_LIBTORCH_CPP_MODEL_H

#include <iostream>
#include <torch/torch.h>
#include<torch/nn.h>

class model {
    // 网络结构体
    struct MNIST_CNNImpl : torch::nn::Module {
        MNIST_CNNImpl():
                // 输入==> torch.Size([64, 1, 28, 28])
                conv_layer1(torch::nn::Conv2dOptions(/*in_channels=*/1, /*out_channels=*/32, /*kernel_size=*/5)),
                bn2d_layer1(torch::nn::BatchNorm2d(32)), // 输出==> torch.Size([64, 32, 24, 24])

                // 输入==> torch.Size([64, 32, 24, 24])
                conv_layer2(torch::nn::Conv2dOptions(/*in_channels=*/32, /*out_channels=*/64, /*kernel_size=*/5)),
                bn2d_layer2(torch::nn::BatchNorm2d(64)), // 输出==> torch.Size([64, 64, 16, 16])

                // 输入==> torch.Size([64, 64, 16, 16])
                conv_layer3(torch::nn::Conv2dOptions(/*in_channels=*/64, /*out_channels=*/64, /*kernel_size=*/5)), // 输出==> torch.Size([64, 64, 8, 8])

                // 输入==> torch.Size([64, 64, 8, 8])
                conv_layer4(torch::nn::Conv2dOptions(/*in_channels=*/64, /*out_channels=*/64, /*kernel_size=*/3)), // 输出==> torch.Size([64, 64, 4, 4])

                // 输入==> torch.Size([64, 64, 4, 4])
                flatten_fc(torch::nn::Flatten()),
                linear_fc_1(torch::nn::LinearOptions(64*4*4, 256)),
                linear_fc_2(torch::nn::LinearOptions(256, 10)) // 输出==> torch.Size([64, 10])

        // 模块注册
        {
            register_module("conv_layer1", conv_layer1);
            register_module("bn2d_layer1", bn2d_layer1);

            register_module("conv_layer2", conv_layer2);
            register_module("bn2d_layer2", bn2d_layer2);

            register_module("conv_layer3", conv_layer3);

            register_module("conv_layer4", conv_layer4);

            register_module("flatten_fc", flatten_fc);
            register_module("linear_fc_1", linear_fc_1);
            register_module("linear_fc_2", linear_fc_2);
        }

        torch::Tensor forward(torch::Tensor x) {
            // 第一层
            x = conv_layer1->forward(x);
            x = torch::relu(bn2d_layer1->forward(x));

            // 第二层
            x = conv_layer2->forward(x);
            x = torch::max_pool2d(torch::relu(bn2d_layer2->forward(x)), { 5, 5 }, 1, { 2, 2 }, 2, false);

            // 第三层
            x = torch::max_pool2d(torch::relu(conv_layer3->forward(x)), { 5, 5 }, 1, { 2, 2 }, 2, false);

            // 第四层
            x = torch::max_pool2d(torch::relu(conv_layer4->forward(x)), { 3, 3 }, 1, { 1, 1 }, 2, false);

            // 第五层
            x = torch::relu(flatten_fc->forward(x));
            x = torch::relu(linear_fc_1->forward(x));
            x = linear_fc_2->forward(x);
            return torch::log_softmax(x, /*dim=*/1);
        }

        // 第一层提取特征，放大特征==> Conv2d -> BatchNorm2d -> ReLU
        torch::nn::Conv2d conv_layer1{nullptr};
        torch::nn::BatchNorm2d bn2d_layer1{nullptr};

        // 第二层提取特征，放大特征==> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        torch::nn::Conv2d conv_layer2{nullptr};
        torch::nn::BatchNorm2d bn2d_layer2{nullptr};

        // 第三层减少信息冗余==> Conv2d -> ReLU -> MaxPool2d
        torch::nn::Conv2d conv_layer3{nullptr};

        // 第四层减少信息冗余==> Conv2d -> ReLU -> MaxPool2d
        torch::nn::Conv2d conv_layer4{nullptr};

        // 第五层全连接，转换成10分类==> Flatten -> ReLU -> Linear -> ReLU -> Linear
        torch::nn::Flatten flatten_fc{nullptr};
        torch::nn::Linear linear_fc_1{nullptr};
        torch::nn::Linear linear_fc_2{nullptr};
    };
    TORCH_MODULE(MNIST_CNN); // 导出网络

private:
    // 设置学习率
    const double_t learning_rate = 1e-3;
    // batch_size设置
    static const int64_t batch_size = 64;
    // 初始化滤波轮数
    const int64_t filter_epoch = 5;
    // 初始化训练轮数
    const int64_t epoch = 10;
public:
    // ------- model自定义函数 -------
    model(); // 构造器
    void net_test();
    torch::DeviceType device_pick();
    void train_model();
    void test_model();
    void save_model(MNIST_CNN& net);
    torch::serialize::InputArchive load_model();
    // ------- model自定义函数 -------
};
#endif //MNIST_LIBTORCH_CPP_MODEL_H
