import ArgumentParser
import Foundation
import CreateMLToolkit

struct TrainSound: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sound",
        abstract: "Train a sound classification model"
    )

    @Argument(help: "Directory containing labeled subdirectories of audio files")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .long, help: "Overlap factor for audio analysis (0.0-1.0)")
    var overlap: Double = 0.5

    @Option(name: .long, help: "Directory containing validation audio files")
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

        let trainer = SoundClassifierTrainer()

        var parameters = SoundClassifierTrainer.TrainingParameters(overlapFactor: overlap)

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

    private func printResult(_ result: SoundClassifierTrainer.TrainingResult) {
        print("")
        print("Training Complete!")
        print("=".repeated(50))
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

    private func printJSON(_ result: SoundClassifierTrainer.TrainingResult) {
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
