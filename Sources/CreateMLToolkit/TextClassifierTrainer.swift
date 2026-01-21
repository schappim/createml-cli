import Foundation
import CreateML
import CoreML
import NaturalLanguage

/// Trains text classification models
public class TextClassifierTrainer {

    public enum Algorithm {
        case maxEnt
        case transferLearning
    }

    public struct TrainingParameters {
        public var algorithm: Algorithm
        public var validationData: URL?

        public init(
            algorithm: Algorithm = .maxEnt,
            validationData: URL? = nil
        ) {
            self.algorithm = algorithm
            self.validationData = validationData
        }
    }

    public struct TrainingResult {
        public let modelURL: URL
        public let trainingAccuracy: Double
        public let validationAccuracy: Double?
        public let trainingDuration: TimeInterval
        public let classLabels: [String]
    }

    public init() {}

    /// Train a text classifier from a JSON or CSV file
    /// - Parameters:
    ///   - trainingDataURL: JSON or CSV file with "text" and "label" columns
    ///   - outputURL: Where to save the trained .mlmodel
    ///   - textColumn: Name of the text column
    ///   - labelColumn: Name of the label column
    ///   - parameters: Training parameters
    ///   - progressHandler: Called with progress updates
    /// - Returns: Training result with metrics
    public func train(
        trainingDataURL: URL,
        outputURL: URL,
        textColumn: String = "text",
        labelColumn: String = "label",
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters = TrainingParameters(),
        progressHandler: ((String) -> Void)? = nil
    ) throws -> TrainingResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = try MLDataTable(contentsOf: trainingDataURL)

        progressHandler?("Found \(trainingData.rows.count) training examples...")

        let algorithmType: MLTextClassifier.ModelAlgorithmType
        switch parameters.algorithm {
        case .maxEnt:
            algorithmType = .maxEnt(revision: nil)
        case .transferLearning:
            // Use dynamic embedding for transfer learning
            algorithmType = .transferLearning(.dynamicEmbedding, revision: nil)
        }

        let modelParameters = MLTextClassifier.ModelParameters(algorithm: algorithmType)

        progressHandler?("Training text classifier using \(algorithmName(parameters.algorithm))...")

        let classifier = try MLTextClassifier(
            trainingData: trainingData,
            textColumn: textColumn,
            labelColumn: labelColumn,
            parameters: modelParameters
        )

        let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
        var validationAccuracy: Double? = nil
        let valError = classifier.validationMetrics.classificationError
        if !valError.isNaN {
            validationAccuracy = (1.0 - valError) * 100
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Text classifier trained with CreateML CLI",
            version: "1.0"
        )

        try classifier.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        // Get unique labels from the data
        var classLabels: [String] = []
        let labelCol = trainingData[labelColumn]
        for i in 0..<trainingData.rows.count {
            if let value = labelCol[i].stringValue {
                if !classLabels.contains(value) {
                    classLabels.append(value)
                }
            }
        }
        classLabels.sort()

        return TrainingResult(
            modelURL: outputURL,
            trainingAccuracy: trainingAccuracy,
            validationAccuracy: validationAccuracy,
            trainingDuration: trainingDuration,
            classLabels: classLabels
        )
    }

    private func algorithmName(_ algorithm: Algorithm) -> String {
        switch algorithm {
        case .maxEnt:
            return "Maximum Entropy"
        case .transferLearning:
            return "Transfer Learning"
        }
    }
}
