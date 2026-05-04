using UnrealBuildTool;

public class PedestrianRuntime : ModuleRules
{
	public PedestrianRuntime(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new[]
			{
				"Core",
				"CoreUObject",
				"DeveloperSettings",
				"Engine",
				"Projects",
				"AeroSemanticRuntime",
				"AeroPedNavSemantic"
			}
		);
	}
}
