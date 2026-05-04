// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Framework/Commands/Commands.h"
#include "TwinFabricEditorStyle.h"

class FTwinFabricEditorCommands : public TCommands<FTwinFabricEditorCommands>
{
public:

	FTwinFabricEditorCommands()
		: TCommands<FTwinFabricEditorCommands>(TEXT("TwinFabricEditor"), NSLOCTEXT("Contexts", "TwinFabricEditor", "TwinFabricEditor Plugin"), NAME_None, FTwinFabricEditorStyle::GetStyleSetName())
	{
	}

	// TCommands<> interface
	virtual void RegisterCommands() override;

public:
	TSharedPtr< FUICommandInfo > PluginAction;
};
