import ArgumentParser
import Foundation
import CreateMLToolkit

struct TrainTabular: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tabular",
        abstract: "Train a tabular classification or regression model"
    )

    @Argument(help: "CSV or JSON file with training data")
    var trainingData: String

    @Option(name: .shortAndLong, help: "Output path for the trained .mlmodel file")
    var output: String

    @Option(name: .shortAndLong, help: "Name of the target column to predict")
    var target: String

    @Option(name: .long, help: "Model type: classifier or regressor")
    var type: String = "classifier"

    @Option(name: .long, help: "Algorithm: auto, randomforest, boostedtree, decisiontree, linear, logistic")
    var algorithm: String = "auto"

    @Option(name: .long, help: "Max tree depth (for tree algorithms)")
    var maxDepth: Int?

    @Option(name: .long, help: "Max iterations (for forest/boosted algorithms)")
    var maxIterations: Int?

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

        let trainer = TabularTrainer()

        let algo: TabularTrainer.TrainingParameters.Algorithm
        switch algorithm.lowercased() {
        case "randomforest", "rf":
            algo = .randomForest(maxDepth: maxDepth, maxIterations: maxIterations)
        case "boostedtree", "boosted", "bt":
            algo = .boostedTree(maxDepth: maxDepth, maxIterations: maxIterations)
        case "decisiontree", "dt":
            algo = .decisionTree(maxDepth: maxDepth)
        case "linear", "linearregression":
            algo = .linearRegression
        case "logistic", "logisticregression":
            algo = .logisticRegression
        default:
            algo = .automatic
        }

        let parameters = TabularTrainer.TrainingParameters(
            targetColumn: target,
            algorithm: algo
        )

        if type.lowercased() == "regressor" || type.lowercased() == "regression" {
            let result = try trainer.trainRegressor(
                trainingDataURL: trainingURL,
                outputURL: outputURL,
                author: author,
                description: description,
                parameters: parameters,
                progressHandler: json ? nil : { print($0) }
            )

            if json {
                printRegressorJSON(result)
            } else {
                printRegressorResult(result)
            }
        } else {
            let result = try trainer.trainClassifier(
                trainingDataURL: trainingURL,
                outputURL: outputURL,
                author: author,
                description: description,
                parameters: parameters,
                progressHandler: json ? nil : { print($0) }
            )

            if json {
                printClassifierJSON(result)
            } else {
                printClassifierResult(result)
            }
        }
    }

    private func printClassifierResult(_ result: TabularTrainer.ClassifierResult) {
        print("")
        print("Training Complete!")
        print("=".repeated(50))
        print("")
        print("Model saved to: \(result.modelURL.path)")
        print("")
        print("Metrics:")
        print("  Training accuracy:   \(String(format: "%.2f", result.trainingAccuracy))%")
        if let valAcc = result.validationAccuracy {
            print("  Validation accuracy: \(String(format: "%.2f", valAcc))%")
        }
        print("  Training duration:   \(String(format: "%.2f", result.trainingDuration))s")
    }

    private func printRegressorResult(_ result: TabularTrainer.RegressorResult) {
        print("")
        print("Training Complete!")
        print("=".repeated(50))
        print("")
        print("Model saved to: \(result.modelURL.path)")
        print("")
        print("Metrics:")
        print("  Training RMSE:       \(String(format: "%.4f", result.trainingRMSE))")
        if let valRMSE = result.validationRMSE {
            print("  Validation RMSE:     \(String(format: "%.4f", valRMSE))")
        }
        print("  Training duration:   \(String(format: "%.2f", result.trainingDuration))s")
    }

    private func printClassifierJSON(_ result: TabularTrainer.ClassifierResult) {
        var output: [String: Any] = [
            "modelPath": result.modelURL.path,
            "trainingAccuracy": result.trainingAccuracy,
            "trainingDurationSeconds": result.trainingDuration
        ]
        if let valAcc = result.validationAccuracy {
            output["validationAccuracy"] = valAcc
        }

        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let jsonString = String(data: data, encoding: .utf8) {
            print(jsonString)
        }
    }

    private func printRegressorJSON(_ result: TabularTrainer.RegressorResult) {
        var output: [String: Any] = [
            "modelPath": result.modelURL.path,
            "trainingRMSE": result.trainingRMSE,
            "trainingDurationSeconds": result.trainingDuration
        ]
        if let valRMSE = result.validationRMSE {
            output["validationRMSE"] = valRMSE
        }

        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let jsonString = String(data: data, encoding: .utf8) {
            print(jsonString)
        }
    }
}
