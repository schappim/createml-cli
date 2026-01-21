import ArgumentParser
import Foundation
import CreateMLToolkit

struct TrainText: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Train a text classification model"
    )

    @Argument(help: "CSV or JSON file with text and label columns")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .long, help: "Name of the text column")
    var textColumn: String = "text"

    @Option(name: .long, help: "Name of the label column")
    var labelColumn: String = "label"

    @Option(name: .long, help: "Algorithm: maxent or transfer")
    var algorithm: String = "maxent"

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
            throw ValidationError("Training data file not found: \(trainingData)")
        }

        let trainer = TextClassifierTrainer()

        let algo: TextClassifierTrainer.Algorithm
        switch algorithm.lowercased() {
        case "transfer", "transferlearning":
            algo = .transferLearning
        default:
            algo = .maxEnt
        }

        let parameters = TextClassifierTrainer.TrainingParameters(algorithm: algo)

        let result = try trainer.train(
            trainingDataURL: trainingURL,
            outputURL: outputURL,
            textColumn: textColumn,
            labelColumn: labelColumn,
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

    private func printResult(_ result: TextClassifierTrainer.TrainingResult) {
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

    private func printJSON(_ result: TextClassifierTrainer.TrainingResult) {
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
