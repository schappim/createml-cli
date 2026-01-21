import ArgumentParser
import Foundation
import CreateMLToolkit

struct TrainObjectDetect: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "object-detect",
        abstract: "Train an object detection model"
    )

    @Argument(help: "Directory containing annotations.json and images")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .long, help: "Maximum training iterations")
    var iterations: Int = 500

    @Option(name: .long, help: "Batch size for training")
    var batchSize: Int = 8

    @Option(name: .long, help: "Directory containing validation data")
    var validation: String?

    @Option(name: .long, help: "Model author")
    var author: String?

    @Option(name: .long, help: "Model description")
    var description: String?

    @Flag(name: .shortAndLong, help: "Output results as JSON")
    var json: Bool = false

    func run() throws {
        let trainingURL = URL(fileURLWithPath: trainingData)
        let outputURL = URL(fileURLWithPath: output)

        guard FileManager.default.fileExists(atPath: trainingData) else {
            throw ValidationError("Training data directory not found: \(trainingData)")
        }

        let trainer = ObjectDetectorTrainer()

        var parameters = ObjectDetectorTrainer.TrainingParameters(
            maxIterations: iterations,
            batchSize: batchSize
        )

        if let validationPath = validation {
            parameters.validationData = URL(fileURLWithPath: validationPath)
        }

        let result = try trainer.train(
            trainingDataURL: trainingURL,
            outputURL: outputURL,
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

    private func printResult(_ result: ObjectDetectorTrainer.TrainingResult) {
        print("")
        print("Training Complete!")
        print("=".repeated(50))
        print("")
        print("Model saved to: \(result.modelURL.path)")
        print("")
        print("Classes: \(result.classLabels.joined(separator: ", "))")
        print("")
        print("Metrics (mAP @ IoU 0.5):")
        print("  Training mAP:        \(String(format: "%.2f", result.trainingMAP * 100))%")
        if let valMAP = result.validationMAP {
            print("  Validation mAP:      \(String(format: "%.2f", valMAP * 100))%")
        }
        print("  Training duration:   \(String(format: "%.2f", result.trainingDuration))s")
    }

    private func printJSON(_ result: ObjectDetectorTrainer.TrainingResult) {
        var output: [String: Any] = [
            "modelPath": result.modelURL.path,
            "trainingMAP": result.trainingMAP,
            "trainingDurationSeconds": result.trainingDuration,
            "classLabels": result.classLabels
        ]
        if let valMAP = result.validationMAP {
            output["validationMAP"] = valMAP
        }

        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let jsonString = String(data: data, encoding: .utf8) {
            print(jsonString)
        }
    }
}
