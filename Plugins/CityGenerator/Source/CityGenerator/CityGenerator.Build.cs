// Copyright Epic Games, Inc. All Rights Reserved.

using System.IO;
using UnrealBuildTool;

public class CityGenerator : ModuleRules
{
    private string ModulePath
    {
        get { return ModuleDirectory; }
    }
    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModulePath, "../../ThirdParty/")); }
    }
    public CityGenerator(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        bEnableUndefinedIdentifierWarnings = false;
        bUseRTTI = true;
        bEnableExceptions = true;
        bUsePrecompiled = true;
        PublicDefinitions.Add("CGAL_CORE_USE_BOOST_BACKEND");
        string BoostVersion = "1_80_0";
        string BoostVersionDir = "boost-" + BoostVersion;
        string BoostPath = Path.Combine(Target.UEThirdPartySourceDirectory, "Boost", BoostVersionDir);
        string BoostIncludePath = Path.Combine(BoostPath, "include");
        PublicIncludePaths.Add(BoostIncludePath);
        PublicIncludePaths.AddRange(
            new string[] {
                Path.Combine(ModuleDirectory, "../../../BinaryGDAL/Public/"),
                Path.Combine(ModuleDirectory, "../../../BinaryGDAL/ThirdParty/include/"),
                Path.Combine(ModuleDirectory, "Private/thirdparty/"),
                 Path.Combine(ModuleDirectory, "Private/core/"),
                 Path.Combine(ModuleDirectory, "Private/config/"),
                 Path.Combine(ModuleDirectory, "Private/processor/"),
            }
            );


        PrivateIncludePaths.AddRange(
            new string[] {
                 Path.Combine(ModuleDirectory, "Private/thirdparty/"),
                 Path.Combine(ModuleDirectory, "Private/core/"),
                 Path.Combine(ModuleDirectory, "Private/config/"),
                 Path.Combine(ModuleDirectory, "Private/processor/"),
 }
            );

        //aliyun oss
        PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "aliyun-oss/include"));
        string x64Path = Path.Combine(ThirdPartyPath, "aliyun-oss/third_party/lib/x64");
        // Add the import library
        PublicAdditionalLibraries.Add(Path.Combine(ThirdPartyPath, "aliyun-oss/lib", "alibabacloud-oss-cpp-sdk.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(x64Path, "libcurl.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(x64Path, "libeay32.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(x64Path, "ssleay32.lib"));

        // copy dll
        var ModuleBinaryFiles = Directory.GetFiles(x64Path, "*", SearchOption.AllDirectories);
        string prefix = Path.Combine(ModuleDirectory, "../../Binaries/Win64");
        if (!Directory.Exists(prefix))
            Directory.CreateDirectory(prefix);
        foreach (var BinaryFile in ModuleBinaryFiles)
        {
            string DependencyName = BinaryFile.Substring(x64Path.Length);
            string BinariesPath = prefix + DependencyName;

            if (!BinariesPath.Contains(".dll"))
                continue;

            if (!File.Exists(BinariesPath))
            {
                File.Copy(BinaryFile, BinariesPath, true);
            }
        }

        // Ensure that the DLL is staged along with the executable
        RuntimeDependencies.Add("$(PluginDir)/Binaries/Win64/libcurl.dll");
        RuntimeDependencies.Add("$(PluginDir)/Binaries/Win64/libeay32.dll");
        RuntimeDependencies.Add("$(PluginDir)/Binaries/Win64/ssleay32.dll");
        RuntimeDependencies.Add("$(PluginDir)/Binaries/Win64/zlibwapi.dll");

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
				// ... add other public dependencies that you statically link with here ...
			}
            );


        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
                "BinaryGDAL",
                "RenderCore",
                "Projects",
                "ProceduralMeshComponent",
                "PCG",
                "HTTP",
                "Json",
                "UnrealEd",
                "AssetTools",
                "MeshDescription",
                "StaticMeshDescription",
                "WebBrowser"
				// ... add private dependencies that you statically link with here ...	
			}
            );


        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );

        if (Target.IsInPlatformGroup(UnrealPlatformGroup.Windows))
        {
            AddEngineThirdPartyPrivateStaticDependencies(Target,
                "Boost"
            );
        }
    }
}
