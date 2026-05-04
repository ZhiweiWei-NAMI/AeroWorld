using UnrealBuildTool;

public class AeroWeatherRender : ModuleRules
{
	public AeroWeatherRender(ReadOnlyTargetRules Target) : base(Target)
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
				"AirSim",
				"Json",
				"JsonUtilities"
			}
		);
	}
}
