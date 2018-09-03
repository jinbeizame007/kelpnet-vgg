using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace VGG
{
    class VGG
    {
        //ミニバッチの数
        const int BATCH_DATA_COUNT = 64;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 781; // = 50000 / 64

        //性能評価時のデータ数
        const int TEACH_DATA_COUNT = 1000;

        public static void Main(string[] args)
        {
            //Cifar-10のデータを用意する
            Console.WriteLine("CIFAR Data Loading...");
            CifarData cifarData = new CifarData();

            //platformIdは、OpenCL・GPUの導入の記事に書いてある方法でご確認ください
            Weaver.Initialize(ComputeDeviceTypes.Gpu, platformId: 1, deviceIndex: 0);

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                /* 最初の4層の畳み込み層を削除
                new Convolution2D (3, 64, 3, pad: 1, gpuEnable: true),
                new ReLU (),
                new Convolution2D (64, 64, 3, pad: 1, gpuEnable: true),
                new ReLU (),
                new MaxPooling(2, 2, gpuEnable: true),

                new Convolution2D (64, 128, 3, pad: 1, gpuEnable: true),
                new ReLU (),
                new Convolution2D (128, 128, 3, pad: 1, gpuEnable: true),
                new ReLU (),
                new MaxPooling(2, 2, gpuEnable: true),
                */

                // (3, 32, 32)
                new Convolution2D(3, 64, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new Convolution2D(64, 64, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new Convolution2D(64, 64, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new MaxPooling(2, 2, gpuEnable: true),

                // (64, 16, 16)
                new Convolution2D(64, 128, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new Convolution2D(128, 128, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new Convolution2D(128, 128, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new MaxPooling(2, 2, gpuEnable: true),

                // (128, 8, 8)
                new Convolution2D(128, 128, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new Convolution2D(128, 128, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new Convolution2D(128, 128, 3, pad: 1, gpuEnable: true),
                new ReLU(),
                new MaxPooling(2, 2, gpuEnable: true),

                // (128, 4, 4)
                new Linear(128 * 4 * 4, 1024, gpuEnable: true),
                new ReLU(),
                new Dropout(0.5),
                new Linear(1024, 1024, gpuEnable: true),
                new ReLU(),
                new Dropout(0.5),
                new Linear(1024, 10, gpuEnable: true)
            );

            //optimizerを宣言
            nn.SetOptimizer(new Adam());

            Console.WriteLine("Training Start...");

            // epoch
            for (int epoch = 1; epoch < 10; epoch++)
            {
                Console.WriteLine("\nepoch " + epoch);

                //全体での誤差を集計
                Real totalLoss = 0;
                long totalLossCount = 0;

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {

                    //Console.WriteLine ("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);

                    //訓練データからランダムにデータを取得
                    TestDataSet datasetX = cifarData.GetRandomXSet(BATCH_DATA_COUNT);

                    //バッチ学習を並列実行する
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());

                    totalLoss += sumLoss;
                    totalLossCount++;

                    //結果出力
                    Console.WriteLine("total loss " + totalLoss / totalLossCount);
                    Console.WriteLine("local loss " + sumLoss);

                    //50回バッチを動かしたら精度をテストする
                    if (i % 50 == 0)
                    {
                        Console.WriteLine("step: " + i + " Testing...");

                        //テストデータからランダムにデータを取得
                        TestDataSet datasetY = cifarData.GetRandomYSet(TEACH_DATA_COUNT);

                        //テストを実行
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}