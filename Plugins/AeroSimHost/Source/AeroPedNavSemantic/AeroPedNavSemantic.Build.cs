using UnrealBuildTool;

public class AeroPedNavSemantic : ModuleRules
{
	public AeroPedNavSemantic(ReadOnlyTargetRules Target) : base(Target)
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
				"Projects"
			}
		);
	}
}
