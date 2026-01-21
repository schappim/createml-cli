import ArgumentParser
import Foundation
import CreateMLToolkit

struct TrainWordTag: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "word-tag",
        abstract: "Train a word tagging model (NER, POS tagging, etc.)"
    )

    @Argument(help: "JSON file with tokens and labels arrays")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .long, help: "Name of the tokens column")
    var tokenColumn: String = "tokens"

    @Option(name: .long, help: "Name of the labels column")
    var labelColumn: String = "labels"

    @Option(name: .long, help: "Path to validation data JSON file")
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
            throw ValidationError("Training data file not found: \(trainingData)")
        }

        let trainer = WordTaggerTrainer()

        var parameters = WordTaggerTrainer.TrainingParameters()
        if let validationPath = validation {
            parameters.validationData = URL(fileURLWithPath: validationPath)
        }

        let result = try trainer.train(
            trainingDataURL: trainingURL,
            outputURL: outputURL,
            tokenColumn: tokenColumn,
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

    private func printResult(_ result: WordTaggerTrainer.TrainingResult) {
        print("")
        print("Training Complete!")
        print("=".repeated(50))
        print("")
        print("Model saved to: \(result.modelURL.path)")
        print("")
        print("Tags: \(result.tagLabels.joined(separator: ", "))")
        print("")
        print("Metrics:")
        print("  Training accuracy:   \(String(format: "%.2f", result.trainingAccuracy))%")
        if let valAcc = result.validationAccuracy {
            print("  Validation accuracy: \(String(format: "%.2f", valAcc))%")
        }
        print("  Training duration:   \(String(format: "%.2f", result.trainingDuration))s")
    }

    private func printJSON(_ result: WordTaggerTrainer.TrainingResult) {
        var output: [String: Any] = [
            "modelPath": result.modelURL.path,
            "trainingAccuracy": result.trainingAccuracy,
            "trainingDurationSeconds": result.trainingDuration,
            "tagLabels": result.tagLabels
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
