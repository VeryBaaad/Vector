plugins { alias(libs.plugins.agp.lib) }

android {
    buildFeatures { aidl = true }

    buildTypes { release { isMinifyEnabled = false } }

    sourceSets {
        named("main") {
            java.srcDir("src/main/java")
            aidl.srcDir("src/main/aidl")
        }
    }

    aidlPackagedList += "org/matrix/vector/ipc/LoadedModule.aidl"
    namespace = "org.matrix.vector.daemonservice"
}

dependencies {
    compileOnly(libs.androidx.annotation)
    compileOnly(libs.libxposed.annotation)
    compileOnly(projects.hiddenapi.stubs)
    compileOnly(libs.libxposed.service)
    compileOnly(libs.libxposed.interfaces)
}
