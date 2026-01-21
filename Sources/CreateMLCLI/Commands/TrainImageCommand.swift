import ArgumentParser
import Foundation
import CreateMLToolkit
import CreateML

struct TrainImage: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image",
        abstract: "Train an image classification model"
    )

    @Argument(help: "Directory containing labeled subdirectories of images")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .shortAndLong, help: "Name for the model")
    var name: String = "ImageClassifier"

    @Option(name: .long, help: "Maximum training iterations")
    var iterations: Int = 25

    @Option(name: .long, help: "Directory containing validation images")
    var validation: String?

    @Option(name: .long, help: "Model author")
    var author: String?

    @Option(name: .long, help: "Model description")
    var description: String?

    @Flag(name: .long, help: "Disable image augmentation")
    var noAugmentation: Bool = false

    @Flag(name: .shortAndLong, help: "Output results as JSON")
    var json: Bool = false

    func run() throws {
        let trainingURL = URL(fileURLWithPath: trainingData)
        let outputURL = URL(fileURLWithPath: output)

        guard FileManager.default.fileExists(atPath: trainingData) else {
            throw ValidationError("Training data directory not found: \(trainingData)")
        }

        let trainer = ImageClassifierTrainer()

        var augmentationOptions: MLImageClassifier.ImageAugmentationOptions = []
        if !noAugmentation {
            augmentationOptions = [.crop, .rotation, .blur, .exposure, .noise, .flip]
        }

        var parameters = ImageClassifierTrainer.TrainingParameters(
            maxIterations: iterations,
            augmentationOptions: augmentationOptions
        )

        if let validationPath = validation {
            parameters.validationData = URL(fileURLWithPath: validationPath)
        }

        let result = try trainer.train(
            trainingDataURL: trainingURL,
            outputURL: outputURL,
            modelName: name,
            author: author,
            description: description,
            parameters: parameters,
            progressHandler: json ? nil : { print($0) }
        )

        if json {
            printJSON(result)
        } else {
            printResult(result)
        }
    }

    private func printResult(_ result: ImageClassifierTrainer.TrainingResult) {
        print("")
        print("Training Complete!")
        print("=" .repeated(50))
        print("")
        print("Model saved to: \(result.modelURL.path)")
        print("")
        print("Classes: \(result.classLabels.joined(separator: ", "))")
        print("")
        print("Metrics:")
        print("  Training accuracy:   \(String(format: "%.2f", result.trainingAccuracy))%")
        if let valAcc = result.validationAccuracy {
            print("  Validation accuracy: \(String(format: "%.2f", valAcc))%")
        }
        print("  Training duration:   \(String(format: "%.2f", result.trainingDuration))s")
    }

    private func printJSON(_ result: ImageClassifierTrainer.TrainingResult) {
        var output: [String: Any] = [
            "modelPath": result.modelURL.path,
            "trainingAccuracy": result.trainingAccuracy,
            "trainingDurationSeconds": result.trainingDuration,
            "classLabels": result.classLabels
        ]
        if let valAcc = result.validationAccuracy {
            output["validationAccuracy"] = valAcc
        }

        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let jsonString = String(data: data, encoding: .utf8) {
            print(jsonString)
        }
    }
}

extension String {
    func repeated(_ count: Int) -> String {
        return String(repeating: self, count: count)
    }
}
