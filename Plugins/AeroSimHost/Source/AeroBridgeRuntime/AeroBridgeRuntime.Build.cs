using UnrealBuildTool;

public class AeroBridgeRuntime : ModuleRules
{
	public AeroBridgeRuntime(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new[]
			{
				"Core",
				"CoreUObject",
				"DeveloperSettings",
				"Engine",
				"Json",
				"JsonUtilities",
				"PedestrianRuntime"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new[]
			{
				"Projects",
				"AirSim",
				"RenderCore",
				"RHI",
				"AeroSemanticRuntime",
				"AeroSceneSync",
				"AeroAssetPlacement",
				"AeroPedNavSemantic",
				"AeroWeatherRender",
				"SumoImporter"
			}
		);
	}
}
