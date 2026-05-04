using UnrealBuildTool;

public class AeroSceneSync : ModuleRules
{
	public AeroSceneSync(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new[]
			{
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
				"AeroAssetPlacement",
				"AeroSemanticRuntime"
			}
		);
	}
}
