import ArgumentParser
import Foundation

@main
struct CreateMLCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "createml",
        abstract: "Train Core ML models from the command line",
        version: "1.2.0",
        subcommands: [
            TrainImage.self,
            TrainText.self,
            TrainSound.self,
            TrainTabular.self,
            TrainObjectDetect.self,
            TrainWordTag.self,
            TrainRecommend.self
        ]
    )
}
