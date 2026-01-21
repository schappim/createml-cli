import Foundation
import CreateML
import CoreML

/// Trains image classification models
public class ImageClassifierTrainer {

    public struct TrainingParameters {
        public var maxIterations: Int
        public var validationData: URL?
        public var augmentationOptions: MLImageClassifier.ImageAugmentationOptions

        public init(
            maxIterations: Int = 25,
            validationData: URL? = nil,
            augmentationOptions: MLImageClassifier.ImageAugmentationOptions = [.crop, .rotation, .blur, .exposure, .noise, .flip]
        ) {
            self.maxIterations = maxIterations
            self.validationData = validationData
            self.augmentationOptions = augmentationOptions
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

    /// Train an image classifier from a directory of labeled images
    /// - Parameters:
    ///   - trainingDataURL: Directory containing subdirectories for each class
    ///   - outputURL: Where to save the trained .mlmodel
    ///   - modelName: Name for the model
    ///   - parameters: Training parameters
    ///   - progressHandler: Called with progress updates
    /// - Returns: Training result with metrics
    public func train(
        trainingDataURL: URL,
        outputURL: URL,
        modelName: String,
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters = TrainingParameters(),
        progressHandler: ((String) -> Void)? = nil
    ) throws -> TrainingResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataURL)

        progressHandler?("Configuring training parameters...")

        let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
            featureExtractor: .scenePrint(revision: 2),
            classifier: .logisticRegressor
        )

        var trainingParameters = MLImageClassifier.ModelParameters(
            maxIterations: parameters.maxIterations,
            augmentation: parameters.augmentationOptions,
            algorithm: algorithm
        )

        if let validationURL = parameters.validationData {
            let validationData = MLImageClassifier.DataSource.labeledDirectories(at: validationURL)
            trainingParameters = MLImageClassifier.ModelParameters(
                validation: .dataSource(validationData),
                maxIterations: parameters.maxIterations,
                augmentation: parameters.augmentationOptions,
                algorithm: algorithm
            )
        }

        progressHandler?("Training image classifier (max \(parameters.maxIterations) iterations)...")

        let classifier = try MLImageClassifier(trainingData: trainingData, parameters: trainingParameters)

        let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
        var validationAccuracy: Double? = nil
        let valError = classifier.validationMetrics.classificationError
        if !valError.isNaN {
            validationAccuracy = (1.0 - valError) * 100
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Image classifier trained with CreateML CLI",
            version: "1.0"
        )

        try classifier.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        // Get class labels
        let classLabels = getClassLabels(from: trainingDataURL)

        return TrainingResult(
            modelURL: outputURL,
            trainingAccuracy: trainingAccuracy,
            validationAccuracy: validationAccuracy,
            trainingDuration: trainingDuration,
            classLabels: classLabels
        )
    }

    private func getClassLabels(from url: URL) -> [String] {
        let fileManager = FileManager.default
        guard let contents = try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey]) else {
            return []
        }
        return contents
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .map { $0.lastPathComponent }
            .sorted()
    }
}
