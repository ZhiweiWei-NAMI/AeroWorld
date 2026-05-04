// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

	public class DynamicCityCreatorEx : ModuleRules
	{
		public DynamicCityCreatorEx(ReadOnlyTargetRules Target) : base(Target)
		{
			PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

			PublicDependencyModuleNames.AddRange(
				new string[]
				{
					"Core",
					"CoreUObject",
					"Engine",
					"InputCore",
					"HeadMountedDisplay",
					"EnhancedInput",
					"ApplicationCore",
					"Slate",
					"SlateCore",
					"Json",
					"AeroBridgeRuntime",
					"AeroAssetPlacement",
					"AeroSemanticRuntime",
					"PedestrianRuntime",
					"SumoImporter"
				});
		}
	}
