using UnrealBuildTool;

public class AeroAssetPlacement : ModuleRules
{
	public AeroAssetPlacement(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new[]
			{
				"AssetRegistry",
				"Core",
				"CoreUObject",
				"Engine"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new[]
			{
				"Json",
				"JsonUtilities",
				"Projects",
				"AirSim",
				"AeroSemanticRuntime",
				"PedestrianRuntime"
			}
		);
	}
}
