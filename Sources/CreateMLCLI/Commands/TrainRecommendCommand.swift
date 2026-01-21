import ArgumentParser
import Foundation
import CreateMLToolkit

struct TrainRecommend: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "recommend",
        abstract: "Train a recommendation model using collaborative filtering"
    )

    @Argument(help: "CSV or JSON file with user-item interactions")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .long, help: "Name of the user column")
    var userColumn: String = "user"

    @Option(name: .long, help: "Name of the item column")
    var itemColumn: String = "item"

    @Option(name: .long, help: "Name of the rating column (optional, for explicit ratings)")
    var ratingColumn: String?

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

        let trainer = RecommenderTrainer()

        let parameters = RecommenderTrainer.TrainingParameters(
            userColumn: userColumn,
            itemColumn: itemColumn,
            ratingColumn: ratingColumn
        )

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

    private func printResult(_ result: RecommenderTrainer.TrainingResult) {
        print("")
        print("Training Complete!")
        print("=".repeated(50))
        print("")
        print("Model saved to: \(result.modelURL.path)")
        print("")
        print("Metrics:")
        if let rmse = result.trainingRMSE {
            print("  Training RMSE:       \(String(format: "%.4f", rmse))")
        }
        if let valRMSE = result.validationRMSE {
            print("  Validation RMSE:     \(String(format: "%.4f", valRMSE))")
        }
        print("  Training duration:   \(String(format: "%.2f", result.trainingDuration))s")
    }

    private func printJSON(_ result: RecommenderTrainer.TrainingResult) {
        var output: [String: Any] = [
            "modelPath": result.modelURL.path,
            "trainingDurationSeconds": result.trainingDuration
        ]
        if let rmse = result.trainingRMSE {
            output["trainingRMSE"] = rmse
        }
        if let valRMSE = result.validationRMSE {
            output["validationRMSE"] = valRMSE
        }

        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let jsonString = String(data: data, encoding: .utf8) {
            print(jsonString)
        }
    }
}
