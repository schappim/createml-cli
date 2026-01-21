// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "createml-cli",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "createml", targets: ["CreateMLCLI"]),
        .library(name: "CreateMLToolkit", targets: ["CreateMLToolkit"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "CreateMLCLI",
            dependencies: [
                "CreateMLToolkit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .target(
            name: "CreateMLToolkit",
            dependencies: []
        ),
        .testTarget(
            name: "CreateMLToolkitTests",
            dependencies: ["CreateMLToolkit"]
        )
    ]
)
