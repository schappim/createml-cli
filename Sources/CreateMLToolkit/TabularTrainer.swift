import Foundation
import CreateML
import CoreML

/// Trains tabular classification and regression models
public class TabularTrainer {

    public enum ModelType {
        case classifier
        case regressor
    }

    public struct TrainingParameters {
        public var targetColumn: String
        public var featureColumns: [String]?
        public var algorithm: Algorithm
        public var validationData: URL?

        public enum Algorithm {
            case automatic
            case randomForest(maxDepth: Int?, maxIterations: Int?)
            case boostedTree(maxDepth: Int?, maxIterations: Int?)
            case decisionTree(maxDepth: Int?)
            case linearRegression  // Regressor only
            case logisticRegression  // Classifier only
        }

        public init(
            targetColumn: String,
            featureColumns: [String]? = nil,
            algorithm: Algorithm = .automatic,
            validationData: URL? = nil
        ) {
            self.targetColumn = targetColumn
            self.featureColumns = featureColumns
            self.algorithm = algorithm
            self.validationData = validationData
        }
    }

    public struct ClassifierResult {
        public let modelURL: URL
        public let trainingAccuracy: Double
        public let validationAccuracy: Double?
        public let trainingDuration: TimeInterval
        public let featureImportance: [String: Double]
    }

    public struct RegressorResult {
        public let modelURL: URL
        public let trainingRMSE: Double
        public let validationRMSE: Double?
        public let trainingDuration: TimeInterval
        public let featureImportance: [String: Double]
    }

    public init() {}

    /// Train a tabular classifier from a CSV or JSON file
    public func trainClassifier(
        trainingDataURL: URL,
        outputURL: URL,
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters,
        progressHandler: ((String) -> Void)? = nil
    ) throws -> ClassifierResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = try MLDataTable(contentsOf: trainingDataURL)

        progressHandler?("Found \(trainingData.rows.count) training examples...")

        let featureColumns = parameters.featureColumns ?? trainingData.columnNames.filter { $0 != parameters.targetColumn }

        progressHandler?("Training tabular classifier on \(featureColumns.count) features...")

        let classifier: MLClassifier

        switch parameters.algorithm {
        case .automatic:
            classifier = try MLClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .randomForest(let maxDepth, let maxIterations):
            var params = MLRandomForestClassifier.ModelParameters()
            if let depth = maxDepth { params.maxDepth = depth }
            if let iterations = maxIterations { params.maxIterations = iterations }
            let rfClassifier = try MLRandomForestClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn, parameters: params)
            classifier = try MLClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .boostedTree(let maxDepth, let maxIterations):
            var params = MLBoostedTreeClassifier.ModelParameters()
            if let depth = maxDepth { params.maxDepth = depth }
            if let iterations = maxIterations { params.maxIterations = iterations }
            let btClassifier = try MLBoostedTreeClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn, parameters: params)
            classifier = try MLClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .decisionTree(let maxDepth):
            var params = MLDecisionTreeClassifier.ModelParameters()
            if let depth = maxDepth { params.maxDepth = depth }
            let dtClassifier = try MLDecisionTreeClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn, parameters: params)
            classifier = try MLClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .logisticRegression:
            let lrClassifier = try MLLogisticRegressionClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
            classifier = try MLClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
        default:
            classifier = try MLClassifier(trainingData: trainingData, targetColumn: parameters.targetColumn)
        }

        let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
        var validationAccuracy: Double? = nil
        let valError = classifier.validationMetrics.classificationError
        if !valError.isNaN {
            validationAccuracy = (1.0 - valError) * 100
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Tabular classifier trained with CreateML CLI",
            version: "1.0"
        )

        try classifier.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        return ClassifierResult(
            modelURL: outputURL,
            trainingAccuracy: trainingAccuracy,
            validationAccuracy: validationAccuracy,
            trainingDuration: trainingDuration,
            featureImportance: [:]
        )
    }

    /// Train a tabular regressor from a CSV or JSON file
    public func trainRegressor(
        trainingDataURL: URL,
        outputURL: URL,
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters,
        progressHandler: ((String) -> Void)? = nil
    ) throws -> RegressorResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = try MLDataTable(contentsOf: trainingDataURL)

        progressHandler?("Found \(trainingData.rows.count) training examples...")

        let featureColumns = parameters.featureColumns ?? trainingData.columnNames.filter { $0 != parameters.targetColumn }

        progressHandler?("Training tabular regressor on \(featureColumns.count) features...")

        let regressor: MLRegressor

        switch parameters.algorithm {
        case .automatic:
            regressor = try MLRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .randomForest(let maxDepth, let maxIterations):
            var params = MLRandomForestRegressor.ModelParameters()
            if let depth = maxDepth { params.maxDepth = depth }
            if let iterations = maxIterations { params.maxIterations = iterations }
            regressor = try MLRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .boostedTree(let maxDepth, let maxIterations):
            var params = MLBoostedTreeRegressor.ModelParameters()
            if let depth = maxDepth { params.maxDepth = depth }
            if let iterations = maxIterations { params.maxIterations = iterations }
            regressor = try MLRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .decisionTree(let maxDepth):
            var params = MLDecisionTreeRegressor.ModelParameters()
            if let depth = maxDepth { params.maxDepth = depth }
            regressor = try MLRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
        case .linearRegression:
            let lrRegressor = try MLLinearRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
            regressor = try MLRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
        default:
            regressor = try MLRegressor(trainingData: trainingData, targetColumn: parameters.targetColumn)
        }

        let trainingRMSE = regressor.trainingMetrics.rootMeanSquaredError
        var validationRMSE: Double? = nil
        let valRMSE = regressor.validationMetrics.rootMeanSquaredError
        if !valRMSE.isNaN {
            validationRMSE = valRMSE
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Tabular regressor trained with CreateML CLI",
            version: "1.0"
        )

        try regressor.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        return RegressorResult(
            modelURL: outputURL,
            trainingRMSE: trainingRMSE,
            validationRMSE: validationRMSE,
            trainingDuration: trainingDuration,
            featureImportance: [:]
        )
    }
}
