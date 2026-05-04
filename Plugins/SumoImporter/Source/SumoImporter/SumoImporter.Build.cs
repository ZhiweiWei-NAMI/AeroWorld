using UnrealBuildTool;

public class SumoImporter : ModuleRules
{
	public SumoImporter(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"XmlParser",
				"Json"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new[]
			{
				"Projects"
			}
		);
	}
}
